import copy
import os
import time
from tqdm import tqdm
import torch
from loss import calc_loss, MultitaskUncertaintyLoss
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer, patience, num_epochs, loss_function, accuracy_metric,  lr_scheduler=None, start_epoch=1):
        self.model = model
        self.dataloader = dataloaders
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.best_loss = 1e15
        self.phases = ["train", "val"]
        self.best_model = []
        self.warm_up = 5
        self.iter_num = 0        
        self.alpha = 10
        self.base_lr = self.optimizer.param_groups[-1]['lr']
        self.max_iterations = self.num_epochs * len(self.dataloader['train'])
        if accuracy_metric in ['dice_score', 'dice_score_mc']:
            self.best_val_score = 0
        else:
            self.best_val_score = 1e15
        self.batch_size = batch_size
        self.output_save_dir = output_save_dir
        self.dtype = dtype
        self.device = device
        self.loss_function = loss_function
        self.accuracy_metric = accuracy_metric
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_score_list = []
        self.model_type = model_type
        self.train_loss_list_1 = []
        self.val_loss_list_1 = []

        self.train_loss_list_2 = []
        self.val_loss_list_2 = []
        self.early_stop_counter = 0
        self.meanTimePerEpoch = 0

        self.save_dir_model = os.path.join(self.output_save_dir, 'models/')
        os.makedirs(self.save_dir_model, exist_ok=True)

    def plot_loss_functions(self, name):
        plt.figure(figsize=(8, 4))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim(0, 1)
        plt.plot(np.arange(len(self.train_loss_list)),
                 self.train_loss_list, label='train loss')
        plt.plot(np.arange(len(self.val_loss_list)),
                 self.val_loss_list, label='val loss')
        plt.plot(np.arange(len(self.val_score_list)),
                 self.val_score_list, label='val accuracy', color='red')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_save_dir, '{}.png'.format(name)))
        plt.cla()

        if self.train_loss_list_1:
            plt.figure(figsize=(8, 4))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(np.arange(len(self.train_loss_list_1)),
                     self.train_loss_list_1, label='train loss')
            plt.plot(np.arange(len(self.val_loss_list_1)),
                     self.val_loss_list_1, label='val loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(
                self.output_save_dir, '{}.png'.format('bce')))
            plt.cla()

        if self.train_loss_list_2:
            plt.figure(figsize=(8, 4))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(np.arange(len(self.train_loss_list_2)),
                     self.train_loss_list_2, label='train loss')
            plt.plot(np.arange(len(self.val_loss_list_2)),
                     self.val_loss_list_2, label='val loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(
                self.output_save_dir, '{}.png'.format('mse')))
            plt.cla()

    def train(self):
        if self.model_type in ['single' ,'TransUnet','regression', 'regression_t', 'attention']:
            if self.loss_function in ['TopoCount', 'TopoCount2', 'TopoLoss', 'TopoLoss2','MyTopoLoss1']:
                self.singe_train_wup()
            else:
                self.singe_train()
        elif self.model_type in ['multi_task', 'multi_task_reg', 'multi_task_regTU']:
            if self.loss_function == 'multi_task_loss':
                self.multi_task_uc_train()
            elif self.loss_function == 'multi_task_loss_ratio':
                self.multi_task_trainRatio()
            else:
                self.multi_task_train()
        else:
            raise ValueError('Invalid model_type "%s"' % self.model_type)

    def singe_train_wup(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")
        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        for epoch in range(self.start_epoch, self.num_epochs+1):
            # dice_loss_list = []
            # hd_loss_list = []
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in self.phases:
                epoch_loss = 0.0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, labels in tbar:
                        tbar.set_description(f"Epoch {epoch}")
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        labels = labels.to(
                            self.device).type(self.dtype)

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            output_mask = self.model(inputs)
                            if self.model_type in ['regression', 'regression_t']:
                                output_mask = F.relu(output_mask)
                                
                            if epoch < 5:
                                loss = calc_loss(output_mask, labels,
                                                loss_type='dice_bce_mc')
                            else:
                                loss = calc_loss(output_mask, labels,
                                loss_type=self.loss_function)

                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                # dice_loss_list.append(loss_dc.item())
                                # hd_loss_list.append(loss_hd.item())
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                                if self.lr_scheduler:
                                    lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = lr_
                                self.iter_num = self.iter_num + 1
                                epoch_loss += loss.item()

                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step, memory=mem)
                            else:
                                epoch_loss += loss.item()                                
                                val_score += loss.item()                           
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_score/(batch_step)), memory=mem)

                epoch_loss /= batch_step
                # deep copy the model
                if phase == 'val':
                    val_score /= batch_step
                    self.val_loss_list.append(epoch_loss)
                    self.val_score_list.append(val_score)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))
                    file.write("\n")

                    if val_score < self.best_val_score and epoch>5:
                        self.early_stop_counter = 0
                        self.best_val_score = val_score
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, model_name))
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, 'best.pt'))
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter > self.patience:
                        print("Early stopping")
                        file.write("Early stopping")
                        file.write("\n")
                        print('Best val loss: {:4f}'.format(self.best_loss))
                        print('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write(
                            'Best val loss: {:4f}'.format(self.best_loss))
                        file.write('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write("\n")
                        file.close()
                        # load best model weights
                        self.model.load_state_dict(self.best_model)
                        self.plot_loss_functions('total')

                        return self.model
                else:
                    # mean_hd = sum(hd_loss_list)/len(hd_loss_list)
                    # mead_dice = sum(dice_loss_list)/len(dice_loss_list)
                    # gamma = mean_hd/mead_dice
                    self.train_loss_list.append(epoch_loss)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

                    torch.save(self.model.state_dict(), os.path.join(
                        self.save_dir_model, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        self.plot_loss_functions('total')
        # load best model weights
        self.model.load_state_dict(self.best_model)
        return self.model
    
    def singe_train(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")
        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'
        totaltime = 0
        for epoch in range(self.start_epoch, self.num_epochs+1):
            # dice_loss_list = []
            # hd_loss_list = []
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in self.phases:
                epoch_loss = 0.0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                    since = time.time()
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, labels in tbar:
                        tbar.set_description(f"Epoch {epoch}")
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        labels = labels.to(
                            self.device).type(self.dtype)

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            output_mask = self.model(inputs)
                            if self.model_type in ['regression', 'regression_t']:
                                output_mask = F.relu(output_mask)
                            loss = calc_loss(output_mask, labels,
                                             loss_type=self.loss_function)

                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                # dice_loss_list.append(loss_dc.item())
                                # hd_loss_list.append(loss_hd.item())
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                                if self.lr_scheduler:
                                    lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = lr_
                                self.iter_num = self.iter_num + 1
                                epoch_loss += loss.item()

                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step, memory=mem)

                            else:
                                epoch_loss += loss.item()
                                val_score += calc_loss(output_mask, labels,
                                                       loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_score.item()/(batch_step)), memory=mem)

                epoch_loss /= batch_step
                # deep copy the model
                if phase == 'val':
                    val_score /= batch_step
                    self.val_loss_list.append(epoch_loss)
                    self.val_score_list.append(val_score.item())
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))
                    file.write("\n")

                    if val_score < self.best_val_score:
                        self.early_stop_counter = 0
                        self.best_val_score = val_score
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, model_name))
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, 'best.pt'))
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter > self.patience:
                        print("Early stopping")
                        file.write("Early stopping")
                        file.write("\n")
                        print('Best val loss: {:4f}'.format(self.best_loss))
                        print('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write(
                            'Best val loss: {:4f}'.format(self.best_loss))
                        file.write('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write("\n")
                        file.close()
                        # load best model weights
                        self.model.load_state_dict(self.best_model)
                        self.plot_loss_functions('total')

                        return self.model
                else:
                    # mean_hd = sum(hd_loss_list)/len(hd_loss_list)
                    # mead_dice = sum(dice_loss_list)/len(dice_loss_list)
                    # gamma = mean_hd/mead_dice
                    time_elapsed = time.time() - since
                    print('Training Time for this epoch: {:.0f}m {:.0f}s\n'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    file.write('Training Time for this epoch: {:.0f}m {:.0f}s\n'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    file.write("\n")
                    self.train_loss_list.append(epoch_loss)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")
                    totaltime += time_elapsed
                    self.meanTimePerEpoch = totaltime/epoch
                    print('Curent mean training time per epoch: {:.0f}m {:.0f}s\n'.format(
                        self.meanTimePerEpoch // 60, self.meanTimePerEpoch % 60))
                    file.write('Curent mean training time per epoch: {:.0f}m {:.0f}s\n'.format(
                        self.meanTimePerEpoch // 60, self.meanTimePerEpoch % 60))                    
                    torch.save(self.model.state_dict(), os.path.join(
                        self.save_dir_model, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        self.plot_loss_functions('total')
        # load best model weights
        self.model.load_state_dict(self.best_model)
        return self.model

    def multi_task_train(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        for epoch in range(self.start_epoch, self.num_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, labels in tbar:
                        label1, label2 = labels 
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label1 = label1.to(
                            self.device).type(self.dtype)
                        label2 = label2.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            
                            output1, output2 = self.model(inputs)
                            output1 = torch.nn.functional.relu(output1)
                            output2 = torch.nn.functional.relu(output2)
                                                      
                            loss1 = calc_loss(output1, label1,
                                              loss_type=self.loss_function)
                            loss2 = calc_loss(output2, label2,
                                              loss_type=self.loss_function)

                            loss = loss1 + loss2
                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                if self.lr_scheduler:
                                    lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = lr_
                                self.iter_num = self.iter_num + 1                               
                                epoch_loss += loss.item()
                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step,  memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                            else:
                                epoch_loss += loss.item()
                                tbar.set_postfix(loss=epoch_loss/batch_step, memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':
                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if epoch_loss < self.best_val_score:
                        self.early_stop_counter = 0
                        self.best_val_score = epoch_loss
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, model_name))
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, 'best.pt'))
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter > self.patience:
                        print("Early stopping")
                        file.write("Early stopping")
                        file.write("\n")
                        print('Best val loss: {:4f}'.format(self.best_loss))
                        print('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write(
                            'Best val loss: {:4f}'.format(self.best_loss))
                        file.write('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write("\n")
                        file.close()
                        # load best model weights
                        self.model.load_state_dict(self.best_model)
                        self.plot_loss_functions('total')

                        return self.model

                else:
                    self.train_loss_list.append(epoch_loss)
                    self.train_loss_list_1.append(loss1_current_epoch)
                    self.train_loss_list_2.append(loss2_current_epoch)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

                    torch.save(self.model.state_dict(), os.path.join(
                        self.save_dir_model, 'last_epoch.pt'))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        self.plot_loss_functions('total')
        # load best model weights
        self.model.load_state_dict(self.best_model)
        return self.model

    def multi_task_uc_train(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        log_var_task1 = torch.zeros((1,), requires_grad=True)
        log_var_task2 = torch.zeros((1,), requires_grad=True)
        params = ([p for p in self.model.parameters()] +
                  [log_var_task1] + [log_var_task2])
        loss_combiner = MultitaskUncertaintyLoss()

        self.optimizer = optim.Adam(params, lr=5e-4)

        for epoch in range(self.start_epoch, self.num_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, labels in tbar:
                        label1, label2 = labels 
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label1 = label1.to(
                            self.device).type(self.dtype)
                        label2 = label2.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            
                            output1, output2 = self.model(inputs)
                            
                            output1 = torch.nn.functional.relu(output1)
                            output2 = torch.nn.functional.relu(output2)
                                
                            loss1 = calc_loss(output1, label1,
                                              loss_type='mse')
                            loss2 = calc_loss(output2, label2,
                                              loss_type='mse')


                            # Multitask with unceartanity loss
                            loss = loss_combiner(
                                [loss1, loss2], [log_var_task1, log_var_task2], [True, True])
                            
                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                if self.lr_scheduler:
                                    lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = lr_
                                self.iter_num = self.iter_num + 1
                                epoch_loss += loss.item()
                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step,  memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                            else:
                                epoch_loss += loss.item()
                                tbar.set_postfix(loss=epoch_loss/batch_step, memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':

                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if epoch_loss < self.best_val_score:
                        self.early_stop_counter = 0
                        self.best_val_score = epoch_loss
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, model_name))
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, 'best.pt'))
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter > self.patience:
                        print("Early stopping")
                        file.write("Early stopping")
                        file.write("\n")
                        print('Best val loss: {:4f}'.format(self.best_loss))
                        print('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write(
                            'Best val loss: {:4f}'.format(self.best_loss))
                        file.write('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write("\n")
                        file.close()
                        # load best model weights
                        self.model.load_state_dict(self.best_model)
                        self.plot_loss_functions('total')

                        return self.model

                else:
                    std_1 = torch.exp(log_var_task1)**0.5
                    std_2 = torch.exp(log_var_task2)**0.5
                    print([std_1.item(), std_2.item()])
                    self.train_loss_list.append(epoch_loss)
                    self.train_loss_list_1.append(loss1_current_epoch)
                    self.train_loss_list_2.append(loss2_current_epoch)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

                    torch.save(self.model.state_dict(), os.path.join(
                        self.save_dir_model, 'last_epoch.pt'))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        self.plot_loss_functions('total')
        # load best model weights
        self.model.load_state_dict(self.best_model)
        return self.model

    def multi_task_trainRatio(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        for epoch in range(self.start_epoch, self.num_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                part1 = 0
                part2 = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, labels in tbar:
                        label1, label2 = labels 
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label1 = label1.to(
                            self.device).type(self.dtype)
                        label2 = label2.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            
                            output1, output2 = self.model(inputs)
                            output1 = torch.nn.functional.relu(output1)
                            output2 = torch.nn.functional.relu(output2)
                                                      
                            loss1 = calc_loss(output1, label1,
                                              loss_type='mse')
                            loss2 = calc_loss(output2, label2,
                                              loss_type='mse')

                            cellCountGt_immune = torch.sum(label1,axis=(1,2))
                            cellCountPred_immune = torch.sum(output1.squeeze(1),axis=(1,2))

                            cellCountGt_other = torch.sum(label2,axis=(1,2))
                            cellCountPred_other = torch.sum(output2.squeeze(1),axis=(1,2))

                            ratioGT = cellCountGt_immune/(cellCountGt_other+cellCountGt_immune)
                            ratioPred = cellCountPred_immune/(cellCountPred_other+cellCountPred_immune)
                            ratioAccuracy = torch.mean(abs(ratioGT-ratioPred))
                           
                            if epoch>5:
                                loss = (loss1 + loss2)*(1+(10*ratioAccuracy))
                            else:
                                loss = loss1 + loss2
                            loss = loss.to(self.device)
                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                part1 += loss1.detach().item() + loss2.detach().item()
                                part2 += (loss1.detach().item() + loss2.detach().item())*ratioAccuracy
                                self.optimizer.step()
                                if self.lr_scheduler:
                                    lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = lr_
                                self.iter_num = self.iter_num + 1
                                
                                epoch_loss += loss.item()
                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step,  memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                            else:
                                epoch_loss += loss.item()
                                tbar.set_postfix(loss=epoch_loss/batch_step, memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':
                    if epoch <= 5:
                        continue
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_score)

                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if epoch_loss < self.best_val_score:
                        self.early_stop_counter = 0
                        self.best_val_score = epoch_loss
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, model_name))
                        torch.save(self.best_model, os.path.join(
                            self.save_dir_model, 'best.pt'))
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter > self.patience:
                        print("Early stopping")
                        file.write("Early stopping")
                        file.write("\n")
                        print('Best val loss: {:4f}'.format(self.best_loss))
                        print('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write(
                            'Best val loss: {:4f}'.format(self.best_loss))
                        file.write('Best val score: {:4f}'.format(
                            self.best_val_score))

                        file.write("\n")
                        file.close()
                        # load best model weights
                        self.model.load_state_dict(self.best_model)
                        self.plot_loss_functions('total')

                        return self.model

                else:
                    self.train_loss_list.append(epoch_loss)
                    self.train_loss_list_1.append(loss1_current_epoch)
                    self.train_loss_list_2.append(loss2_current_epoch)
                    part1 /= batch_step 
                    part2 /= batch_step 
                    self.alpha = part1/part2
                    file.write("Alpha on epoch %i: %f" % (epoch, self.alpha))
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

                    torch.save(self.model.state_dict(), os.path.join(
                        self.save_dir_model, 'last_epoch.pt'))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        self.plot_loss_functions('total')
        # load best model weights
        self.model.load_state_dict(self.best_model)
        return self.model
