# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from .conditional_detr import build
from argparse import Namespace

def buildCLTR(args):
    args = Namespace(**args)

    model, criterion, postprocessors = build(args)

    return model, criterion, postprocessors
