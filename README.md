# StackChan workspace

The live project is **[`stackchan-neuromorphic/`](stackchan-neuromorphic/README.md)**: a hybrid robot brain
(vision and hearing networks, a 768-neuron spiking core, a GRU policy, an analogue droid voice) for the
M5Stack StackChan (CoreS3).

- `datasets/`         training data (COCO val2017 + annotations, ESC-50, LibriSpeech dev-clean), extracted
- `firmware-builder/` Dockerfile of the ESP-IDF build image (`stackchan-firmware-builder:local`)
- `.tooling/`         ESP-IDF v5.5.4 toolchain cache used by the Docker build (git-ignored)
- `backups/`          one backup of the current state: `stackchan-neuromorphic-v3-2026-10-04.tgz` + firmware `.bin`
                      (see stackchan-neuromorphic/README.md, Rollback)
