const { execSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const packageRoot = path.join(__dirname, "..");
const ds4Path = path.join(packageRoot, "ds4");

function run(command, options = {}) {
  execSync(command, {
    cwd: packageRoot,
    stdio: "inherit",
    ...options,
  });
}

if (!fs.existsSync(ds4Path)) {
  const packageJson = JSON.parse(
    fs.readFileSync(path.join(packageRoot, "package.json"), "utf8")
  );
  const { repo, commit } = packageJson.ds4 ?? {};

  if (!repo || !commit) {
    console.error("ERROR: ds4.repo and ds4.commit must be set in package.json");
    process.exit(1);
  }

  run(`git clone --depth 1 ${repo} ds4`);
  run(`git fetch --depth 1 origin ${commit}`, { cwd: ds4Path });
  run(`git checkout ${commit}`, { cwd: ds4Path });
} else {
  console.log("ds4 directory already exists, skipping clone...");
}

run("npx node-gyp rebuild");
