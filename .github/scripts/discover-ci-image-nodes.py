#!/usr/bin/env python3
"""Discover the union of nodes allowed by this workflow's ARC job templates.

Read-only: only Kubernetes get/list calls. Readiness, taints and available
capacity deliberately do not narrow the result: they can change while CI runs.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


def requirement_matches(requirement, attributes):
    key = requirement['key']
    operator = requirement['operator']
    values = requirement.get('values', [])
    present = key in attributes
    value = attributes.get(key)
    if operator == 'In':
        return present and value in values
    if operator == 'NotIn':
        return not present or value not in values
    if operator == 'Exists':
        return present
    if operator == 'DoesNotExist':
        return not present
    if operator in ('Gt', 'Lt'):
        if len(values) != 1:
            raise ValueError('Invalid numeric affinity requirement')
        bound = int(values[0])
        if not present:
            return False
        try:
            number = int(value)
        except ValueError:
            return False
        return number > bound if operator == 'Gt' else number < bound
    raise ValueError('Unsupported node affinity operator: ' + operator)


def allowed_by_template(node, spec):
    name = node['metadata']['name']
    labels = node['metadata'].get('labels', {})
    if spec.get('nodeName') and name != spec['nodeName']:
        return False
    if any(labels.get(key) != value for key, value in spec.get('nodeSelector', {}).items()):
        return False
    required = spec.get('affinity', {}).get('nodeAffinity', {}).get(
        'requiredDuringSchedulingIgnoredDuringExecution')
    if required is None:
        return True
    for term in required.get('nodeSelectorTerms', []):
        expressions = term.get('matchExpressions', [])
        fields = term.get('matchFields', [])
        if not expressions and not fields:
            continue
        if any(field['key'] != 'metadata.name' for field in fields):
            raise ValueError('Unsupported node affinity field')
        if all(requirement_matches(item, labels) for item in expressions) and all(
                requirement_matches(item, {'metadata.name': name}) for item in fields):
            return True
    return False


def kubernetes_json(*arguments):
    binary = os.environ.get('CI_KUBECTL') or shutil.which('kubectl')
    if not binary and Path('/public/bin/kubectl').is_file():
        binary = '/public/bin/kubectl'
    if not binary:
        raise RuntimeError('kubectl is required for dynamic ARC node discovery')
    result = subprocess.check_output(
        [binary, '--request-timeout=30s'] + list(arguments) + ['-o', 'json'])
    return json.loads(result)


def discover():
    workflow = yaml.safe_load((Path(__file__).parent.parent / 'workflows/ci.yml').read_text())
    pools = set()
    for job in workflow['jobs'].values():
        if 'container' not in job:
            continue
        pool = job['runs-on']
        if not isinstance(pool, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', pool):
            raise ValueError('ARC jobs must use an explicit runner scale-set name')
        pools.add(pool)
    if not pools:
        raise ValueError('No ARC runner pools found in workflow')
    namespace = os.environ.get('CI_ARC_NAMESPACE', 'arc-runners')
    templates = {}
    for pool in sorted(pools):
        config = kubernetes_json('-n', namespace, 'get', 'configmap', 'job-template-' + pool)
        template = yaml.safe_load(config['data']['content'])
        templates[pool] = template['spec']
    nodes = kubernetes_json('get', 'nodes')['items']
    selected = {}
    counts = {pool: 0 for pool in pools}
    for node in nodes:
        matches = [pool for pool, spec in templates.items() if allowed_by_template(node, spec)]
        if not matches:
            continue
        for pool in matches:
            counts[pool] += 1
        name = node['metadata']['name']
        address = node['metadata'].get('annotations', {}).get('ci.hygon.cn/ssh-host')
        if not address:
            internal = [item['address'] for item in node.get('status', {}).get('addresses', [])
                        if item['type'] == 'InternalIP']
            address = internal[0] if internal else name
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.:-]*', address):
            raise ValueError('Invalid SSH address for node ' + name)
        if address in selected:
            raise ValueError('Duplicate SSH address in node inventory: ' + address)
        selected[address] = name
        print('Image preparation candidate: {} ({}) for {}'.format(
            name, address, ', '.join(sorted(matches))), file=sys.stderr)
    for pool, count in counts.items():
        if not count:
            raise ValueError('No matching nodes discovered for ARC pool ' + pool)
    for address in sorted(selected):
        print(address)


if __name__ == '__main__':
    try:
        discover()
    except Exception as error:
        print('::error::Dynamic ARC node discovery failed: {}. No static node fallback.'.format(error),
              file=sys.stderr)
        sys.exit(1)
