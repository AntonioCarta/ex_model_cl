import re

import yaml


class YAMLConfig:
    def __init__(self, config_file):
        self.config_files = [config_file]
        self._parse_config(config_file)

    def config_name(self):
        s = []
        for el in self.config_files:
            # drop the folder, keep only the filename
            el = el.split('/')[-1].split('.')[0]
            s.append(el)
        return '_'.join(s)

    def update(self, new_config_file):
        """ Parse yaml file """
        self.config_files.append(new_config_file)
        self._parse_config(new_config_file)

    def _parse_config(self, config_file):
        """
        Parse yaml file containing also math notation like 1e-4
        """
        # fix to enable scientific notation
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(config_file, 'r') as f:
            configs = yaml.load(f, Loader=loader)

        for k, v in configs.items():
            self.__dict__[k] = v

    def __str__(self):
        res = 'CONFIG:\n'
        for k, v in self.__dict__.items():
            res += f'\t({k}) -> {v}\n'
        return res

    def __eq__(self, other):
        da = self.__dict__
        db = other.__dict__

        ignore_keys = {'config_file'}
        def check_inclusion_recursive(config_a, config_b):
            for k, v in da.items():
                if k == 'config_files':
                    continue  # we don't care about filenames, they can be different.
                if k in db and db[k] == v:
                    continue
                else:
                    return False
            return True
        return check_inclusion_recursive(da, db) and check_inclusion_recursive(db, da)


__all__ = [
    'YAMLConfig'
]