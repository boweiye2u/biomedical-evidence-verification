import torch
from transformers import BertConfig, BertModel
from retrieval.training import explicit_loss

def test_shared_encoder_gradients_and_optimizer_update():
    torch.manual_seed(123)
    model=BertModel(BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,hidden_dropout_prob=0,attention_probs_dropout_prob=0))
    q={'input_ids':torch.randint(0,32,(2,8))}
    d={'input_ids':torch.randint(0,32,(12,8))}
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    before=model.embeddings.word_embeddings.weight.detach().clone()
    loss,qe,de=explicit_loss(model,q,d)
    qe.retain_grad();de.retain_grad();loss.backward()
    assert torch.isfinite(loss)
    assert qe.grad.abs().sum()>0 and de.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    expected=torch.nn.functional.cross_entropy(torch.einsum('bd,bkd->bk',qe,de)/0.05,torch.zeros(2,dtype=torch.long))
    torch.testing.assert_close(loss,expected)
    optimizer.step()
    assert not torch.equal(before,model.embeddings.word_embeddings.weight)
