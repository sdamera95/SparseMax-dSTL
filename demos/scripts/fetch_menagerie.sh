#!/usr/bin/env bash
# Fetch the pinned MuJoCo Menagerie Panda model into third_party/ (gitignored).
# Run from demos/. The commit is also pinned in sparsemax_diffstl/plants.py.
set -eu
commit=c96a32d28fb5da84da38c1da4d749e7a13212855
dest=third_party/mujoco_menagerie
if [ ! -d "$dest/.git" ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$dest"
fi
git -C "$dest" sparse-checkout set franka_emika_panda
git -C "$dest" fetch -q origin "$commit"
git -C "$dest" -c advice.detachedHead=false checkout -q "$commit"
git -C "$dest" rev-parse HEAD
