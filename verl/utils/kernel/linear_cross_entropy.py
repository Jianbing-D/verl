#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import typing
from . import kernels


class LinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                labels: torch.Tensor,
                reduction: typing.Optional[str] = "mean") -> typing.List[torch.Tensor]:
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            logprobs, entropy, _maximum, _acc, _d_scale =\
                kernels.efficient_entropy_foward(hidden, weight, labels, REDUCTION)

            ctx.save_for_backward(hidden, weight, labels, _maximum, _acc, _d_scale)
            ctx.REDUCTION = REDUCTION

        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> typing.List[torch.Tensor]:
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _acc, _d_scale) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION

            d_hidden, d_weight = kernels.efficient_entropy_backward(dlogprobs, dentropy, hidden, weight, labels,
                                                                    _maximum, _acc, _d_scale, REDUCTION)

        return (d_hidden, d_weight, None, None)


linear_cross_entropy = LinearCrossEntropy.apply
