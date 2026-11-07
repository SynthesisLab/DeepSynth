from argparse import ArgumentParser, Namespace
from typing import List, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from synth import PBE, Task
from synth.nn import (
    DetGrammarPredictorLayer,
    UGrammarPredictorLayer,
    abstractions,
    Task2Tensor,
)
from synth.pbe import IOEncoder
from synth.syntax import UCFG, TTCFG
from synth.syntax.grammars.cfg import CFG


class MyPredictor(nn.Module):
    def __init__(
        self,
        size: int,
        constrained: bool,
        cfgs: Union[List[TTCFG], List[UCFG]],
        variable_probability: float,
        encoding_dimension: int,
        device: str,
        lexicon,
        io_padding: bool = True,
    ) -> None:
        super().__init__()
        layer = UGrammarPredictorLayer if constrained else DetGrammarPredictorLayer
        abstraction = (
            abstractions.ucfg_bigram
            if constrained
            else abstractions.cfg_bigram_without_depth
        )
        self.bigram_layer = layer(
            size,
            cfgs,
            abstraction,
            variable_probability,
        )
        encoder = IOEncoder(encoding_dimension, lexicon, padding=io_padding)
        self.packer = Task2Tensor(
            encoder, nn.Embedding(len(encoder.lexicon), size), size, device=device
        )
        self.rnn = nn.LSTM(size, size, 1)
        self.end = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
        )

    def forward(self, x: List[Task[PBE]]) -> Tensor:
        seq: PackedSequence = self.packer(x)
        _, (y, _) = self.rnn(seq)
        y: Tensor = y.squeeze(0)
        return self.bigram_layer(self.end(y))


def instantiate_predictor(
    parameters: Namespace, cfgs: Union[List[CFG], List[UCFG]], lexicon: List
) -> MyPredictor:
    variable_probability: float = parameters.var_prob
    encoding_dimension: int = parameters.encoding_dimension
    hidden_size: int = parameters.hidden_size
    cpu_only: bool = parameters.cpu
    constrained: bool = parameters.constrained
    device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"

    return MyPredictor(
        hidden_size,
        constrained,
        cfgs,
        variable_probability,
        encoding_dimension,
        device,
        lexicon,
        io_padding=not parameters.no_io_padding,
    ).to(device)


def add_model_choice_arg(parser: ArgumentParser) -> None:
    gg = parser.add_argument_group("model parameters")
    gg.add_argument(
        "-v",
        "--var-prob",
        type=float,
        default=0.2,
        help="variable probability (default: .2)",
    )
    gg.add_argument(
        "-ed",
        "--encoding-dimension",
        type=int,
        default=512,
        help="encoding dimension (default: 512)",
    )
    gg.add_argument(
        "-hd",
        "--hidden-size",
        type=int,
        default=512,
        help="hidden layer size (default: 512)",
    )
    gg.add_argument(
        "--no-io-padding",
        action="store_true",
        default=False,
        help="do not pad the encoding of each example to the encoding dimension",
    )
    gg.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="do not try to run things on cuda",
    )
    gg = parser.add_argument_group("grammar parameters")
    gg.add_argument(
        "--constrained",
        action="store_true",
        default=False,
        help="use unambigous grammar to include constraints in the grammar if available",
    )

    gg.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="maximum depth of grammars used (-1 for infinite, default: 5)",
    )
    gg.add_argument(
        "--ngram",
        type=int,
        default=2,
        choices=[1, 2],
        help="ngram used by grammars (default: 2)",
    )
