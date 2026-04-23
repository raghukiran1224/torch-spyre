# Copyright 2025 The Torch-Spyre Authors.
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

"""FX graph pass to rewrite index_select on expert weights to moe_expert_gather.

Detects the pattern ``torch.index_select(experts, 0, indices)`` where
``experts`` is a 2-D+ weight tensor on Spyre and rewrites it to
``torch.ops.spyre.moe_expert_gather(experts, input_dummy, indices_2d, gate_dummy)``.
"""

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from .constants import DEVICE_NAME

aten = torch.ops.aten

moe_gather_pass = PatternMatcherPass(pass_name="moe_expert_gather_rewrite")


def _is_spyre_tensor_node(node):
    if not isinstance(node, torch.fx.Node):
        return False
    val = node.meta.get("val")
    if val is None or not isinstance(val, torch.Tensor):
        return False
    return getattr(val.device, "type", None) == DEVICE_NAME


@register_graph_pattern(
    CallFunction(aten.index_select.default, Arg(), Arg(), Arg()),
    pass_dict=moe_gather_pass,
)
def _rewrite_index_select_to_moe_gather(
    match: Match,
    experts_node: torch.fx.Node,
    dim_node,
    indices_node: torch.fx.Node,
) -> None:
    dim = dim_node
    if isinstance(dim_node, torch.fx.Node):
        return
    if dim != 0:
        return

    if not _is_spyre_tensor_node(experts_node):
        return
    experts_val = experts_node.meta["val"]
    if experts_val.dim() < 2:
        return

    if not isinstance(indices_node, torch.fx.Node):
        return
    indices_val = indices_node.meta.get("val")
    if indices_val is None or indices_val.dim() != 1:
        return

    node = match.nodes[-1]
    graph = node.graph

    batch = indices_val.shape[0]
    hidden = experts_val.shape[-1]
    dtype = experts_val.dtype
    device = experts_val.device

    with graph.inserting_before(node):
        input_dummy = graph.call_function(
            aten.empty.memory_format,
            args=([batch, hidden],),
            kwargs={"dtype": dtype, "device": device},
        )
        input_dummy.meta["val"] = torch.empty(
            batch, hidden, dtype=dtype, device="meta"
        )

        indices_unsqueeze = graph.call_function(
            aten.unsqueeze.default, args=(indices_node, -1)
        )
        indices_unsqueeze.meta["val"] = torch.empty(
            batch, 1, dtype=indices_val.dtype, device="meta"
        )

        indices_2d = graph.call_function(
            aten.expand.default,
            args=(indices_unsqueeze, [batch, hidden]),
        )
        indices_2d.meta["val"] = torch.empty(
            batch, hidden, dtype=indices_val.dtype, device="meta"
        )

        gate_dummy = graph.call_function(
            aten.ones.default,
            args=([batch, hidden],),
            kwargs={"dtype": dtype, "device": device},
        )
        gate_dummy.meta["val"] = torch.ones(
            batch, hidden, dtype=dtype, device="meta"
        )

        moe_node = graph.call_function(
            torch.ops.spyre.moe_expert_gather.default,
            args=(experts_node, input_dummy, indices_2d, gate_dummy),
        )
        moe_node.meta["val"] = torch.empty(
            batch, hidden, dtype=dtype, device="meta"
        )

    node.replace_all_uses_with(moe_node)
    graph.erase_node(node)
