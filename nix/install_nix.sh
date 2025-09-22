#!/bin/bash

mkdir -m 0755 /nix && chown root /nix
mkdir -m 0755 /etc/nix && chown root /etc/nix
echo "build-users-group =" |  tee -a /etc/nix/nix.conf
bash <(curl -L https://nixos.org/nix/install) --daemon

#  nix-build -A pythonFull '<nixpkgs>'