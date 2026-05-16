# Maintainers

This repository uses Changesets for npm releases of `@lgrammel/ds4-provider`.

## Release Setup

The GitHub release workflow publishes from `main` and expects the repository
secret `NPM_TOKEN` to contain an npm automation token with publish access to the
`@lgrammel/ds4-provider` package.

The package vendors the pinned DS4 source listed in
`packages/ds4-provider/package.json` under the `ds4.commit` field. Published
tarballs include only the DS4 files needed to compile the native addon, not GGUF
model files. Users still need to download a compatible GGUF separately.

## Normal Release Flow

1. Make the code changes.
2. Add a changeset:

   ```sh
   pnpm changeset
   ```

3. Pick the appropriate bump type and write a concise user-facing summary.
4. Validate locally:

   ```sh
   pnpm typecheck
   pnpm --filter @lgrammel/ds4-provider run build
   npm pack --dry-run --json --workspace packages/ds4-provider
   ```

5. Open and merge the feature PR into `main`.
6. The release workflow will create a Changesets version PR.
7. Review the version PR, especially `CHANGELOG.md`, package versions, and the
   package lockfile.
8. Merge the version PR into `main`.
9. The release workflow will publish the package to npm.

## Updating DS4

To update the bundled DS4 source:

1. Choose a commit from `https://github.com/antirez/ds4.git`.
2. Update `ds4.commit` in `packages/ds4-provider/package.json`.
3. Refresh the local vendored checkout:

   ```sh
   pnpm --filter @lgrammel/ds4-provider run vendor:ds4
   ```

4. Build and test the native addon:

   ```sh
   pnpm --filter @lgrammel/ds4-provider run build
   pnpm --filter @lgrammel/ds4-provider test
   ```

5. Run a pack dry-run and confirm the tarball contains `dist`, `native`,
   `binding.gyp`, `scripts`, `ds4/ds4.c`, `ds4/ds4.h`, `ds4/ds4_gpu.h`,
   `ds4/ds4_metal.m`, `ds4/LICENSE`, and `ds4/metal`.

## Manual Publish

Prefer the GitHub workflow. If a manual publish is needed, make sure you are on
the versioned release commit from `main`, authenticated to npm, and then run:

```sh
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm changeset status
pnpm release
```

`pnpm release` builds the package and runs `changeset publish`.
