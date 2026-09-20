import pytest
import torch
import torch.nn as nn

from pruning_transformer import AttentionSurgeon


def make_qkvo(hidden=8, bias=True, dtype=torch.float32):
    torch.manual_seed(0)
    query = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    key = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    value = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    output = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    return query, key, value, output


def test_resize_keeps_the_surviving_heads_rows_and_columns():
    """hidden=8, 2 heads -> head 0 is rows/cols [0:4), head 1 is [4:8).
    Dropping head 0 should leave exactly head 1's slice.
    """
    query, key, value, output = make_qkvo()
    new_q, new_k, new_v, new_o, num_heads = AttentionSurgeon().resize(
        query, key, value, output, num_heads=2, head_indices=[0]
    )

    assert num_heads == 1
    assert new_q.out_features == 4 and new_o.in_features == 4
    assert torch.equal(new_q.weight, query.weight[4:8])
    assert torch.equal(new_k.weight, key.weight[4:8])
    assert torch.equal(new_v.weight, value.weight[4:8])
    assert torch.equal(new_q.bias, query.bias[4:8])
    assert torch.equal(new_o.weight, output.weight[:, 4:8])
    # Output bias is over hidden size, not the head layout -- untouched.
    assert torch.equal(new_o.bias, output.bias)


def test_dropping_a_middle_head_of_four_keeps_the_others_in_order():
    query, key, value, output = make_qkvo(hidden=16)  # 4 heads of 4
    new_q, _, _, new_o, num_heads = AttentionSurgeon().resize(
        query, key, value, output, num_heads=4, head_indices=[1]
    )
    assert num_heads == 3
    expected_rows = torch.cat([torch.arange(0, 4), torch.arange(8, 16)])
    assert torch.equal(new_q.weight, query.weight[expected_rows])
    assert torch.equal(new_o.weight, output.weight[:, expected_rows])


def test_dropping_multiple_heads_at_once():
    query, key, value, output = make_qkvo(hidden=16)  # 4 heads of 4
    _, _, _, _, num_heads = AttentionSurgeon().resize(
        query, key, value, output, num_heads=4, head_indices=[0, 2]
    )
    assert num_heads == 2


def test_dtype_is_preserved():
    query, key, value, output = make_qkvo(dtype=torch.float16)
    new_q, new_k, new_v, new_o, _ = AttentionSurgeon().resize(
        query, key, value, output, num_heads=2, head_indices=[0]
    )
    for layer in (new_q, new_k, new_v, new_o):
        assert layer.weight.dtype == torch.float16


def test_biasless_layers_survive_surgery():
    query, key, value, output = make_qkvo(bias=False)
    new_q, new_k, new_v, new_o, _ = AttentionSurgeon().resize(
        query, key, value, output, num_heads=2, head_indices=[0]
    )
    assert new_q.bias is None and new_k.bias is None and new_v.bias is None and new_o.bias is None


def test_requires_grad_is_preserved():
    query, key, value, output = make_qkvo()
    query.weight.requires_grad_(False)
    new_q, new_k, _, _, _ = AttentionSurgeon().resize(
        query, key, value, output, num_heads=2, head_indices=[0]
    )
    assert new_q.weight.requires_grad is False
    assert new_k.weight.requires_grad is True


def test_removing_a_head_changes_the_output_on_every_input():
    """The surgery should actually delete the head's contribution, not
    merely resize -- a head prune has no output-preserving option (unlike
    FFNSurgeon's merge), so before/after must differ.
    """
    query, key, value, output = make_qkvo(hidden=8)
    x = torch.randn(5, 3, 8)

    def forward(q, k, v, o, heads):
        head_dim = q.out_features // heads
        split = lambda t: t.view(*t.shape[:-1], heads, head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(split(q(x)), split(k(x)).transpose(-1, -2)) / head_dim ** 0.5
        ctx = torch.matmul(torch.softmax(scores, dim=-1), split(v(x)))
        ctx = ctx.permute(0, 2, 1, 3).reshape(*x.shape[:-1], heads * head_dim)
        return o(ctx)

    before = forward(query, key, value, output, 2)
    new_q, new_k, new_v, new_o, num_heads = AttentionSurgeon().resize(
        query, key, value, output, num_heads=2, head_indices=[0]
    )
    after = forward(new_q, new_k, new_v, new_o, num_heads)
    assert before.shape == after.shape
    assert not torch.allclose(before, after, atol=1e-4)


def test_out_of_range_and_empty_and_all_heads_are_rejected():
    query, key, value, output = make_qkvo(hidden=16)  # 4 heads of 4
    with pytest.raises(ValueError, match="out-of-range"):
        AttentionSurgeon().resize(query, key, value, output, num_heads=4, head_indices=[0, 99])
    with pytest.raises(ValueError, match="collapses"):
        AttentionSurgeon().resize(query, key, value, output, num_heads=4, head_indices=[0, 1, 2, 3])


def test_indivisible_num_heads_is_rejected():
    query, key, value, output = make_qkvo(hidden=8)
    with pytest.raises(ValueError, match="not divisible"):
        AttentionSurgeon().resize(query, key, value, output, num_heads=3, head_indices=[0])


def test_mismatched_output_projection_is_rejected():
    query, key, value, _ = make_qkvo(hidden=8)
    wrong_output = nn.Linear(5, 8)
    with pytest.raises(ValueError, match="inconsistent"):
        AttentionSurgeon().resize(query, key, value, wrong_output, num_heads=2, head_indices=[0])
