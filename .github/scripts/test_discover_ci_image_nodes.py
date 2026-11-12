import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'discovery', str(Path(__file__).with_name('discover-ci-image-nodes.py')))
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)


def node(name, labels=None):
    return {'metadata': {'name': name, 'labels': labels or {}},
            'status': {'addresses': [{'type': 'InternalIP', 'address': 'node-' + name}]}}


class DiscoveryTests(unittest.TestCase):
    def test_selector_requires_every_label(self):
        template = {'nodeSelector': {'build': 'true', 'arch': 'x86'}}
        self.assertTrue(discovery.allowed_by_template(node('a', {'build': 'true', 'arch': 'x86'}), template))
        self.assertFalse(discovery.allowed_by_template(node('b', {'build': 'true'}), template))

    def test_required_affinity_uses_or_terms_and_intersects_selector(self):
        template = {'nodeSelector': {'build': 'true'}, 'affinity': {'nodeAffinity': {
            'requiredDuringSchedulingIgnoredDuringExecution': {'nodeSelectorTerms': [
                {'matchExpressions': [{'key': 'pool', 'operator': 'In', 'values': ['a']}]},
                {'matchExpressions': [{'key': 'generation', 'operator': 'Gt', 'values': ['3']}]}]}}}}
        self.assertTrue(discovery.allowed_by_template(node('a', {'build': 'true', 'pool': 'a'}), template))
        self.assertTrue(discovery.allowed_by_template(node('b', {'build': 'true', 'generation': '4'}), template))
        self.assertFalse(discovery.allowed_by_template(node('c', {'generation': '4'}), template))
        self.assertFalse(discovery.allowed_by_template(node('d', {'build': 'true', 'generation': 'bad'}), template))

    def test_negative_and_presence_requirements(self):
        for operator, values, attributes, expected in [
                ('NotIn', ['no'], {}, True), ('NotIn', ['no'], {'x': 'no'}, False),
                ('Exists', [], {}, False), ('Exists', [], {'x': ''}, True),
                ('DoesNotExist', [], {}, True), ('DoesNotExist', [], {'x': ''}, False),
                ('Lt', ['4'], {'x': '3'}, True), ('Lt', ['4'], {}, False)]:
            with self.subTest(operator=operator, attributes=attributes):
                self.assertEqual(discovery.requirement_matches(
                    {'key': 'x', 'operator': operator, 'values': values}, attributes), expected)

    def test_empty_required_term_matches_nothing(self):
        template = {'affinity': {'nodeAffinity': {
            'requiredDuringSchedulingIgnoredDuringExecution': {'nodeSelectorTerms': [{}]}}}}
        self.assertFalse(discovery.allowed_by_template(node('a'), template))

    def test_node_name_field(self):
        template = {'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchFields': [
                {'key': 'metadata.name', 'operator': 'In', 'values': ['a']}]}]}}}}
        self.assertTrue(discovery.allowed_by_template(node('a'), template))
        self.assertFalse(discovery.allowed_by_template(node('b'), template))

    def test_new_matching_node_is_discovered_without_address_inventory(self):
        workflow = 'jobs:\n  build:\n    runs-on: hygon-build\n    container: {image: ci}\n'
        nodes = [node('a', {'build': 'true'}), node('b', {'build': 'true'}),
                 node('new', {'build': 'true'}), node('other', {'build': 'false'})]
        nodes[2]['spec'] = {'unschedulable': True}
        nodes[2]['status']['conditions'] = [{'type': 'Ready', 'status': 'False'}]
        def response(*args):
            if 'configmap' in args:
                return {'data': {'content': 'spec:\n  nodeSelector: {build: "true"}\n'}}
            return {'items': nodes}
        output = io.StringIO()
        with patch.object(Path, 'read_text', return_value=workflow), \
                patch.object(discovery, 'kubernetes_json', side_effect=response), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            discovery.discover()
        self.assertEqual(output.getvalue().splitlines(), ['node-a', 'node-b', 'node-new'])

    def test_unsupported_rule_fails(self):
        with self.assertRaises(ValueError):
            discovery.requirement_matches({'key': 'x', 'operator': 'Unknown'}, {})


if __name__ == '__main__':
    unittest.main()
