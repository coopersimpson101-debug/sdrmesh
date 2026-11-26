# Contributing to sdrmesh

Thanks for your interest! Bug reports, captures, test cases and documentation fixes are all welcome.

## Ground rules

- **Keep it honest.** Every number and plot in the README comes from code in this repository. Simulated results
  are labelled as simulated; hardware results must say which SDR, antenna, gains and environment were used. Never
  fill a hardware table from a simulation.
- **Write the DSP yourself.** The point of the project is that the signal processing is hand-written. NumPy is
  used for arrays, FFTs and convolution; please do not add DSP libraries (GNU Radio, scipy.signal, liquid-dsp) to
  the core package.
- **Tests first.** Every behaviour change needs a test. Run the suite before opening a PR:
  ```bash
  pip install -e ".[test,meshfind,plot]"
  python -m pytest -q
  ```
- **Results.** If a change affects BER/PER, re-run `python tools/run_montecarlo.py` and
  `python docs/assets/make_figures.py`, and commit the regenerated JSON and figures together.
- **Style.** Follow `.editorconfig` (4-space indent, LF line endings); document each function with the maths it
  implements and a reference.
- **Radio rules.** Only transmit where you are licensed or within licence-exempt limits (2.4 GHz ISM at low
  power, 868/915 MHz SRD/ISM duty-cycle rules). Prefer cabled tests with attenuators.

## Reporting bugs and ideas

Use the issue templates. For security problems see [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the MIT License of this repository.
