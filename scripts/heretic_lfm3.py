"""Run Heretic 2.0 against a locally trusted LFM3 Omni checkpoint.

This adapter is experimental: evaluate modified models independently before use.
Install Heretic in a separate environment and pass its options after ``--``.
"""

import argparse
import json
from pathlib import Path
import sys
from types import MethodType
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='Local full-precision LFM3 Omni checkpoint directory')
    parser.add_argument('--trust-model-code', action='store_true',
                        help='Allow execution of custom Python code from this checkpoint')
    parser.add_argument('heretic_args', nargs=argparse.REMAINDER,
                        help='Heretic options, following --')
    args = parser.parse_args()
    target = args.model_dir.resolve(strict=True)
    if not args.trust_model_code:
        parser.error('inspect the checkpoint code and pass --trust-model-code to run it')
    config = json.loads((target / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'lfm3_omni':
        parser.error('expected an LFM3 Omni checkpoint')
    if version('heretic-llm') != '2.0.0':
        parser.error('this experimental adapter requires heretic-llm==2.0.0')
    passthrough = args.heretic_args
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]
    if any(arg == '--model' or arg.startswith('--model=') for arg in passthrough):
        parser.error('use --model-dir to select the checkpoint')

    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationMixin

    for auto_class in (AutoProcessor, AutoModelForImageTextToText):
        original = auto_class.from_pretrained.__func__

        def wrap(inner):
            def load(cls, model, *positional, **kwargs):
                is_target = Path(str(model)).resolve() == target
                if is_target:
                    kwargs['trust_remote_code'] = True
                loaded = inner(cls, model, *positional, **kwargs)
                if is_target and cls is AutoModelForImageTextToText:
                    # Heretic needs the text generation API's logits and hidden states;
                    # LFM3's custom speech sampler uses the same method name.
                    loaded.generate = MethodType(GenerationMixin.generate, loaded)
                    type(loaded)._sample = GenerationMixin._sample
                    # The original greedy generation config fails current save validation.
                    loaded.generation_config.temperature = 1.0
                    loaded.generation_config.top_k = 50
                return loaded
            return classmethod(load)

        auto_class.from_pretrained = wrap(original)

    from heretic.main import main as heretic_main
    sys.argv = ['heretic', '--model', str(target), *passthrough]
    heretic_main()


if __name__ == '__main__':
    main()
