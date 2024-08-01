# contract

<img src="header.png" alt="Neural tokenization" title="Source: Image by Author and generated with MidJourney" width="100%" style="margin: auto;"/>

EVM compiler, using neural networks with a transformer architecture.

More precisely, it ingests JSON formatted sources and outputs EVM bytecode, in binary.

It is built upon:

- the dataset of smart contracts [feedblocks][github-feedblocks]
- the neural encoder [tokun][github-tokun]
- the standard encoder-decoder architecture of transformers

## Resources

### Notebooks

### Articles

## License

Licensed under the [aGPLv3](LICENSE.md).

[github-feedblocks]: https://github.com/apehex/feedblocks
[github-tokun]: https://github.com/apehex/tokun
[solidity-docs-json]: https://docs.soliditylang.org/en/v0.8.26/using-the-compiler.html#compiler-input-and-output-json-description
