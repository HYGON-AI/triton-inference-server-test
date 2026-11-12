const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const {createHash} = require('node:crypto');
const {publish} = require('./publish-ci-image.cjs');

const id = 'sha256:' + 'a'.repeat(64);
const digest = 'sha256:' + 'b'.repeat(64);

async function fixture(t, existing = false) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'triton-publish-test-'));
  t.after(() => fs.rm(directory, {recursive: true, force: true}));
  await fs.writeFile(path.join(directory, 'image-tag'), 'ci:source');
  await fs.writeFile(path.join(directory, 'image-id'), id);
  await fs.writeFile(path.join(directory, 'image.tar.gz'), 'test-archive');
  await fs.writeFile(path.join(directory, 'image.tar.gz.sha256'),
    createHash('sha256').update('test-archive').digest('hex') + '  image.tar.gz');
  const calls = [];
  let packageExists = existing;
  const api = async () => {
    if (!packageExists) throw Object.assign(new Error('missing'), {status: 404});
    return {data: {visibility: 'private'}};
  };
  const core = {setSecret() {}, info() {}, setOutput(key, value) {calls.push(['output', key, value]);}};
  const run = (args) => {
    calls.push(args);
    if (args[0] === 'copy') {packageExists = true; return '';}
    if (args.includes('--format')) return digest;
    return JSON.stringify({config: {digest: id}});
  };
  return {calls, directory, options: {core, run,
    github: {rest: {packages: {getPackageForOrganization: api}}},
    context: {repo: {owner: 'Example', repo: 'Test'}, actor: 'test-user'},
    env: {CI_IMAGE: 'ci:source', CI_IMAGE_ARCHIVE_DIR: directory, GH_TOKEN: 'test-token', RUNNER_TEMP: directory}}};
}

test('publishes the verified archive and outputs a digest usable on any node', async t => {
  const {options, calls, directory} = await fixture(t);
  const result = await publish(options);
  assert.equal(result, `ghcr.io/example/test-ci@${digest}`);
  assert.equal(calls.filter(args => args[0] === 'copy').length, 1);
  assert.equal((await fs.readdir(directory)).some(name => name.startsWith('triton-registry-')), false);
});

test('existing matching registry image skips archive transfer', async t => {
  const {options, calls, directory} = await fixture(t, true);
  await fs.unlink(path.join(directory, 'image.tar.gz'));
  await publish(options);
  assert.equal(calls.some(args => args[0] === 'copy'), false);
});

test('corrupt archive fails before publication', async t => {
  const {options, calls, directory} = await fixture(t);
  await fs.writeFile(path.join(directory, 'image.tar.gz'), 'corrupt');
  await assert.rejects(publish(options), /SHA-256 mismatch/);
  assert.equal(calls.some(args => args[0] === 'copy'), false);
});

test('existing wrong image ID is never overwritten', async t => {
  const {options, calls} = await fixture(t, true);
  options.run = args => {
    calls.push(args);
    return JSON.stringify({config: {digest: 'sha256:' + 'c'.repeat(64)}});
  };
  await assert.rejects(publish(options), /Registry image ID mismatch/);
  assert.equal(calls.some(args => args[0] === 'copy'), false);
});

test('public package is rejected before any image operation', async t => {
  const {options, calls} = await fixture(t);
  options.github.rest.packages.getPackageForOrganization = async () => ({data: {visibility: 'public'}});
  await assert.rejects(publish(options), /non-private/);
  assert.equal(calls.length, 0);
});
