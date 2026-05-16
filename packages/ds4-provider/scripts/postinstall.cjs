const { execSync } = require("node:child_process");
const path = require("node:path");

const packageRoot = path.join(__dirname, "..");

function run(command) {
  execSync(command, {
    cwd: packageRoot,
    stdio: "inherit",
  });
}

run("node-gyp rebuild");
