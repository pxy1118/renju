# Third-party components

## Rapfi

- Location: `external/rapfi` Git submodule
- Upstream: https://github.com/dhbloo/rapfi
- License: GPL-3.0, as declared by the upstream `Copying.txt`
- Integration: optional external process protocol; no Rapfi object code is
  linked into the MIT-licensed Python application.

The build helper creates `external/rapfi-runtime`, which is ignored by Git and
is not part of this project's distributable Python package. Distributing a
Rapfi binary or a recursive checkout remains subject to Rapfi's GPL-3.0 terms.

## Rapfi Networks

- Location: `external/rapfi/Networks` nested submodule
- Upstream: https://github.com/dhbloo/rapfi-networks
- License: CC0, as declared by that repository

The network files are copied only into the ignored local runtime directory.
