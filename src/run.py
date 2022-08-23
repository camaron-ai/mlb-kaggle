import click
import os
from pathlib import Path
from dotenv import load_dotenv

plugin_folder = Path(os.path.join(os.path.dirname(__file__), 'command'))


class CliManager(click.MultiCommand):
    suffix = ''
    cli_function = 'cli'

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(plugin_folder):
            if filename.startswith('_'):
                continue
            if filename.endswith('.py'):
                # remove the suffix and .py
                rv.append(filename[len(self.suffix):-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        ns = {}
        fn = os.path.join(plugin_folder, f'{self.suffix}{name}.py')
        with open(fn) as f:
            code = compile(f.read(), fn, 'exec')
            eval(code, ns, ns)
        return ns['cli']


cli = CliManager(help='subcommand tool for machine learning')


if __name__ == '__main__':
    load_dotenv()

    cli()
