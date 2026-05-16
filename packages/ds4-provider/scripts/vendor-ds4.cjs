const { execSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const packageRoot = path.join(__dirname, "..");
const packageJsonPath = path.join(packageRoot, "package.json");
const ds4Path = path.join(packageRoot, "ds4");
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
const { repo, commit } = packageJson.ds4 ?? {};

if (!repo || !commit) {
  console.error("ERROR: ds4.repo and ds4.commit must be set in package.json");
  process.exit(1);
}

function run(command, options = {}) {
  execSync(command, {
    cwd: packageRoot,
    stdio: "inherit",
    ...options,
  });
}

function isStandaloneGitCheckout() {
  return fs.existsSync(path.join(ds4Path, ".git"));
}

function hasVendoredSource() {
  return fs.existsSync(path.join(ds4Path, "ds4.c")) &&
    fs.existsSync(path.join(ds4Path, "ds4.h")) &&
    fs.existsSync(path.join(ds4Path, "metal"));
}

function getCurrentCommit() {
  if (!isStandaloneGitCheckout()) {
    return undefined;
  }

  try {
    return execSync("git rev-parse HEAD", {
      cwd: ds4Path,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return undefined;
  }
}

if (!fs.existsSync(ds4Path)) {
  run(`git clone --depth 1 ${repo} ds4`);
} else if (!isStandaloneGitCheckout()) {
  if (!hasVendoredSource()) {
    console.error(`ERROR: ${ds4Path} exists but does not contain DS4 sources`);
    process.exit(1);
  }

  console.log("Using existing vendored ds4 source directory");
  process.exit(0);
}

if (getCurrentCommit() !== commit) {
  run(`git fetch --depth 1 origin ${commit}`, { cwd: ds4Path });
  run(`git checkout --detach ${commit}`, { cwd: ds4Path });
}

console.log(`Vendored ds4 at ${commit}`);
