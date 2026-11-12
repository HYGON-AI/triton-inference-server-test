// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
const fs = require('node:fs/promises');
const {createReadStream} = require('node:fs');
const {createHash} = require('node:crypto');
const {spawnSync} = require('node:child_process');
const path = require('node:path');
const os = require('node:os');

function skopeo(args, stream = false) {
  const result = spawnSync('skopeo', args, {
    encoding: 'utf8', stdio: stream ? 'inherit' : 'pipe', maxBuffer: 4 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || `skopeo exited ${result.status}`);
  return result.stdout || '';
}

async function sha256(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

async function publish({core, github, context, run = skopeo, env = process.env}) {
  const directory = env.CI_IMAGE_ARCHIVE_DIR ||
    '/ci_public/ci-images/triton-inference-server-test-ci/20261111-aicc-g2f732c';
  const sourceImage = (await fs.readFile(path.join(directory, 'image-tag'), 'utf8')).trim();
  const expectedId = (await fs.readFile(path.join(directory, 'image-id'), 'utf8')).trim();
  const archiveSha = (await fs.readFile(path.join(directory, 'image.tar.gz.sha256'), 'utf8')).trim().split(/\s+/)[0];
  if (env.CI_IMAGE !== sourceImage) throw new Error('CI_IMAGE does not match the shared image archive.');
  if (!/^sha256:[a-f0-9]{64}$/.test(expectedId) || !/^[a-f0-9]{64}$/.test(archiveSha)) {
    throw new Error('Invalid shared image metadata.');
  }
  const owner = context.repo.owner.toLowerCase();
  const packageName = context.repo.repo.toLowerCase() + '-ci';
  const registry = `ghcr.io/${owner}/${packageName}`;
  const tagged = `${registry}:sha256-${expectedId.slice(7)}`;
  async function requirePrivate() {
    try {
      const {data} = await github.rest.packages.getPackageForOrganization({
        package_type: 'container', package_name: packageName, org: context.repo.owner,
      });
      if (data.visibility !== 'private') throw new Error('Refusing to publish to a non-private image package.');
      return true;
    } catch (error) {
      if (error.status !== 404) throw error;
      // A package first published by GITHUB_TOKEN is private and linked to this repo.
      return false;
    }
  }
  const packageExists = await requirePrivate();
  const token = env.GH_TOKEN;
  if (!token) throw new Error('The workflow GITHUB_TOKEN is required.');
  core.setSecret(token);
  const authDirectory = await fs.mkdtemp(path.join(env.RUNNER_TEMP || os.tmpdir(), 'triton-registry-'));
  const authFile = path.join(authDirectory, 'auth.json');
  try {
    const encoded = Buffer.from(`${context.actor}:${token}`).toString('base64');
    core.setSecret(encoded);
    await fs.writeFile(authFile, JSON.stringify({auths: {'ghcr.io': {auth: encoded}}}), {mode: 0o600});
    const inspect = () => {
      try {
        return JSON.parse(run(['inspect', '--raw', '--authfile', authFile, `docker://${tagged}`]));
      } catch (error) {
        if (/manifest unknown|name unknown|manifest does not exist/i.test(error.message)) return null;
        throw error;
      }
    };
    let manifest = packageExists ? inspect() : null;
    if (!manifest) {
      const archive = path.join(directory, 'image.tar.gz');
      core.info('Verifying the shared archive before publishing the private CI image.');
      if (await sha256(archive) !== archiveSha) throw new Error('Shared archive SHA-256 mismatch.');
      const source = JSON.parse(run(['inspect', '--raw', `docker-archive:${archive}`]));
      if (source.config?.digest !== expectedId) throw new Error('Shared archive image ID mismatch.');
      run(['copy', '--retry-times', '3', '--format', 'v2s2', '--dest-authfile', authFile,
        `docker-archive:${archive}`, `docker://${tagged}`], true);
      manifest = inspect();
    } else {
      core.info('Matching private CI image already exists; skipping archive transfer.');
    }
    if (manifest?.config?.digest !== expectedId) throw new Error('Registry image ID mismatch.');
    if (!await requirePrivate()) throw new Error('Cannot verify the published package is private.');
    const digest = run(['inspect', '--format', '{{.Digest}}', '--authfile', authFile, `docker://${tagged}`]).trim();
    if (!/^sha256:[a-f0-9]{64}$/.test(digest)) throw new Error('Invalid registry manifest digest.');
    const image = `${registry}@${digest}`;
    core.setOutput('image', image);
    core.info(`ARC jobs will use ${image}; every selected node can pull this exact image.`);
    return image;
  } finally {
    await fs.rm(authDirectory, {recursive: true, force: true});
  }
}

module.exports = {publish, sha256};

if (require.main === module) {
  const token = process.env.GH_TOKEN;
  const [owner, repo] = (process.env.GITHUB_REPOSITORY || '').split('/');
  const core = {
    setSecret(value) { console.log(`::add-mask::${value}`); },
    info(value) { console.log(value); },
    setOutput(key, value) {
      require('node:fs').appendFileSync(process.env.GITHUB_OUTPUT, `${key}=${value}\n`);
    },
  };
  const github = {rest: {packages: {
    async getPackageForOrganization({org, package_type, package_name}) {
      const response = await fetch(`https://api.github.com/orgs/${encodeURIComponent(org)}/packages/${package_type}/${encodeURIComponent(package_name)}`, {
        headers: {Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json',
          'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'triton-ci-image-publisher'},
        signal: AbortSignal.timeout(30000),
      });
      if (!response.ok) throw Object.assign(new Error(`Package API returned HTTP ${response.status}`), {status: response.status});
      return {data: await response.json()};
    },
  }}};
  publish({core, github, context: {repo: {owner, repo}, actor: process.env.GITHUB_ACTOR}})
    .catch(error => { console.error(`::error::${error.message}`); process.exitCode = 1; });
}
