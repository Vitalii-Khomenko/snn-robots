# secrets/

The firmware needs two headers here that are **not** in the repository (real Wi-Fi password, IPs of the owner's devices):

```bash
cp wifi_credentials.example.h wifi_credentials.h    # then edit
cp known_devices.example.h    known_devices.h       # then edit
```

Both real files are git-ignored. Never commit them, post them in screenshots or put them into archives.
