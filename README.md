# Financial Sentiment Model

A GPT-style decoder-only transformer trained from scratch in PyTorch (multi-head self-attention, rotary positional embeddings (RoPE), custom byte-pair encoding tokenizer) on a corpus of financial news headlines and earnings call transcripts, fine-tuned for sentiment classification. The model's sentiment output feeds into a backtesting framework as a trading signal.

## Status

In progress.

## Roadmap

- [ ] Financial text corpus collection (news headlines, earnings transcripts)
- [ ] Custom BPE tokenizer
- [ ] Transformer architecture (multi-head self-attention, RoPE)
- [ ] Pretraining loop
- [ ] Fine-tuning for sentiment classification
- [ ] Integration with backtesting-framework as a trading signal

## Tech Stack

Python, PyTorch

## Dependencies

Feeds into [backtesting-framework](https://github.com/Varish-Sanad/backtesting-framework).
