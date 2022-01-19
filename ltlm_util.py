import pdb
import torch
from torch import nn, Tensor
from typing import Any, Dict


class Config:
    def __init__(self, **kwargs):
        def set_required(name):
            if name not in kwargs:
                raise Exception(f"No {name} provided to config.")
            setattr(self, name, kwargs[name])

        set_required("model_type")
        set_required("num_tm_words")
        set_required("max_seqlen")
        set_required("pad_idx")
        self.beta = kwargs.pop("beta", 1.0)
        self.use_all_bows = kwargs.pop("use_all_bows", False)
        self.eval_false = kwargs.pop("eval_false", False)
        self.reset_hidden = kwargs.pop("reset_hidden", True)


def repackage_hidden(h, reset_hidden):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return torch.zeros_like(h.detach()) if reset_hidden else h.detach()
    else:
        return tuple(repackage_hidden(v, reset_hidden) for v in h)


def compute_loss(
    cf: Dict[str, Any],
    output: Dict[str, Tensor],
    target: Tensor,
    stopwords: Tensor,
    doc_lens: list,
) -> Tensor:
    mask = (target != cf.pad_idx).float()
    num_tokens_per_seq = (target.view(-1, len(doc_lens)) != cf.pad_idx).sum(0)
    num_tokens = num_tokens_per_seq.sum()

    if cf.model_type == "TopicRNN":
        token_criterion = nn.NLLLoss(ignore_index=cf.pad_idx)
        stopword_criterion = nn.BCEWithLogitsLoss(reduction="none")
        stopword_target = stopwords[1:].view(-1).float()
        kl_loss = (
            -0.5
            / doc_lens
            * num_tokens_per_seq
            * torch.sum(
                1
                + output["logsigma_theta"]
                - output["mu_theta"].pow(2)
                - output["logsigma_theta"].exp(),
                dim=-1,
            )
        ).sum() / len(target)
        token_loss = token_criterion(output["token_output"], target)
        stopword_loss = (
            mask * stopword_criterion(output["stopword_output"], stopword_target)
        ).sum() / len(target)

        return {
            "token_loss": token_loss,
            "stopword_loss": stopword_loss,
            "kl_loss": kl_loss,
            "total_loss": token_loss + stopword_loss + cf.beta * kl_loss,
            "num_tokens": num_tokens,
        }
    elif cf.model_type == "VAE":
        token_criterion = nn.NLLLoss(ignore_index=cf.pad_idx)
        kl_loss = (
            -0.5
            / doc_lens
            * torch.sum(
                1
                + output["logsigma_theta"]
                - output["mu_theta"].pow(2)
                - output["logsigma_theta"].exp(),
                dim=-1,
            )
        ).mean()
        token_loss = token_criterion(output["token_output"], target)

        return {
            "token_loss": token_loss,
            "stopword_loss": torch.tensor(0),
            "kl_loss": kl_loss,
            "total_loss": token_loss + cf.beta * kl_loss,
            "num_tokens": num_tokens,
        }
    else:
        criterion = nn.NLLLoss(ignore_index=cf.pad_idx)
        return {
            "total_loss": criterion(output["token_output"], target),
            "num_tokens": num_tokens,
        }


def compute_log_ppl(
    cf: Dict[str, Any],
    output: Dict[str, Tensor],
    target: Tensor,
    oracle: bool = False,
    reduction: str = "mean",
) -> Tensor:
    criterion = nn.NLLLoss(ignore_index=cf.pad_idx, reduction=reduction)

    if cf.model_type == "TopicRNN" and not oracle:
        stopword_prob = torch.sigmoid(output["stopword_output"]).unsqueeze(-1)
        output = (
            stopword_prob * output["token_rnn_output"].exp()
            + (1 - stopword_prob) * output["token_mixed_output"].exp()
        ).log()
    else:
        output = output["token_output"]

    if reduction == "none":
        log_ppl = criterion(output, target).view(cf.max_seqlen, -1).mean(dim=0)
    else:
        log_ppl = criterion(output, target)

    return {
        "log_ppl": log_ppl,
        "num_tokens": (target != cf.pad_idx).sum(),
    }
