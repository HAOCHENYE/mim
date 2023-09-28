# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import importlib
import inspect
import logging
import os
import os.path as osp
import re
from functools import lru_cache
from typing import Callable

import torch.nn
from mmengine import mkdir_or_exist
from mmengine.logging import print_log
from mmengine.model import (
    BaseDataPreprocessor,
    BaseModel,
    BaseModule,
    ImgDataPreprocessor,
)
from mmengine.registry import Registry
from yapf.yapflib.yapf_api import FormatCode

from mim.utils import OFFICIAL_MODULES
from .common import BUILDER_TRANS, REGISTRY_TYPE
from .flatten_func import *  # noqa: F403, F401
from .flatten_func import (
    flatten_model,
    ignore_ast_docstring,
    init_prepare,
    postprocess_super,
    postprocess_top_ast_tree,
)


def format_code(code_text: str):
    """Format the code text with yapf."""
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    try:
        code_text, _ = FormatCode(
            code_text, style_config=yapf_style,
            verify=False)  # TODO: some code couldn't pass the verify
    except:  # noqa: E722
        raise SyntaxError('Failed to format the config file, please '
                          f'check the syntax of: \n{code_text}')

    return code_text


def _get_all_files(directory: str):
    """Get all files of the directory.

    Args:
        directory (str): The directory path.

    Returns:
        List: Return the a list containing all the files in the directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '__init__' not in file and 'registry.py' not in file:
                file_paths.append(os.path.join(root, file))

    return file_paths


def _transfer_to_export_import(file_path: str):
    """Transfer the importfrom path from "downstream repo" to export module.

    Args:
        file_path (str): The path of file needed to be transfer.

    Examples:
        >>> from mmdet.models.detectors.two_stage import TwoStageDetector
        >>> # transfer to below, if "TwoStageDetector" had been exported
        >>> from pack.models.detectors.two_stage import TwoStageDetector
    """
    from mmengine import Registry

    # _module_path_dict is a class attribute,
    # already record all the exported module and their path before
    _module_path_dict = Registry._module_path_dict

    with open(file_path, encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())

    # if the import module have the same name with the object in these file,
    # they import path won't be change
    can_not_change_module = []
    for node in ast_tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            can_not_change_module.append(node.name)

    def check_change_importfrom_node(node: ast.ImportFrom):
        """Check if the ImportFrom node should be changed.

        If the modules in node had already been exported, they will be
        separated and compose a new ast.ImportFrom node with the export
        path as the module path.

        Args:
            node (ast.ImportFrom): ImportFrom node.

        Returns:
            ast.ImportFrom | None: Return a new ast.ImportFrom node
            if one of the module in node had been export else ``None``.
        """
        export_module_path = None
        needed_change_alias = []

        for alias in node.names:
            if alias.name in _module_path_dict.keys(
            ) and alias.name not in can_not_change_module:

                if export_module_path is None:
                    export_module_path = _module_path_dict[alias.name]
                else:
                    assert _module_path_dict[alias.name] == export_module_path,\
                        'There are two module from the same downstream repo,'\
                        " but can't change to the same export path."

                needed_change_alias.append(alias)

        if len(needed_change_alias) != 0:
            for alias in needed_change_alias:
                node.names.remove(alias)

            return ast.ImportFrom(
                module=export_module_path, names=needed_change_alias, level=0)

        return None

    # Naming rules for searching ast syntax tree
    # - node: node of ast.Module
    # - func_sub_node: sub_node of ast.FunctionDef
    # - class_sub_node: sub_node of ast.ClassDef
    # - func_sub_class_sub_node: sub_node ast.FunctionDef in ast.ClassDef

    # record the insert_idx and node needed to be insert for later insert.
    insert_idx_and_node = {}

    insert_idx = 0
    for node in ast_tree.body:

        # search ast.ImportFrom in ast.Module scope
        # ast.Module -> ast.ImportFrom
        if isinstance(node, ast.ImportFrom):
            insert_idx += 1
            temp_node = check_change_importfrom_node(node)
            if temp_node is not None:
                if len(node.names) == 0:
                    ast_tree.body[insert_idx - 1] = temp_node
                else:
                    insert_idx_and_node[insert_idx] = temp_node
                    insert_idx += 1

        elif isinstance(node, ast.Import):
            insert_idx += 1

        else:
            # search ast.ImportFrom in ast.FunctionDef scope
            # ast.Module -> ast.FunctionDef -> ast.ImportFrom
            if isinstance(node, ast.FunctionDef):
                insert_idx = ignore_ast_docstring(node)
                func_need_to_be_removed_nodes = []

                for func_sub_node in node.body:
                    if isinstance(func_sub_node, ast.ImportFrom):
                        temp_node = check_change_importfrom_node(
                            func_sub_node)  # noqa: E501
                        if temp_node is not None:
                            node.body.insert(insert_idx, temp_node)

                        # if importfrom module is empty, the node should be remove  # noqa: E501
                        if len(func_sub_node.names) == 0:
                            func_need_to_be_removed_nodes.append(
                                func_sub_node)  # noqa: E501

                for need_to_be_removed_node in func_need_to_be_removed_nodes:
                    node.body.remove(need_to_be_removed_node)

            # search ast.ImportFrom in ast.ClassDef scope
            # ast.Module -> ast.ClassDef -> ast.ImportFrom
            #                            -> ast.FunctionDef -> ast.ImportFrom
            elif isinstance(node, ast.ClassDef):
                insert_idx = ignore_ast_docstring(node)
                class_need_to_be_removed_nodes = []

                for class_sub_node in node.body:

                    # ast.Module -> ast.ClassDef -> ast.ImportFrom
                    if isinstance(class_sub_node, ast.ImportFrom):
                        temp_node = check_change_importfrom_node(
                            class_sub_node)
                        if temp_node is not None:
                            node.body.insert(insert_idx, temp_node)
                        if len(class_sub_node.names) == 0:
                            class_need_to_be_removed_nodes.append(
                                class_sub_node)

                    # ast.Module -> ast.ClassDef -> ast.FunctionDef -> ast.ImportFrom  # noqa: E501
                    elif isinstance(class_sub_node, ast.FunctionDef):
                        class_sub_insert_idx = ignore_ast_docstring(node)
                        func_need_to_be_removed_nodes = []

                        for func_sub_class_sub_node in class_sub_node.body:
                            if isinstance(func_sub_class_sub_node,
                                          ast.ImportFrom):
                                temp_node = check_change_importfrom_node(
                                    func_sub_class_sub_node)
                                if temp_node is not None:
                                    node.body.insert(class_sub_insert_idx,
                                                     temp_node)
                                if len(func_sub_class_sub_node.names) == 0:
                                    func_need_to_be_removed_nodes.append(
                                        func_sub_class_sub_node)

                        for need_to_be_removed_node in func_need_to_be_removed_nodes:  # noqa: E501
                            class_sub_node.body.remove(need_to_be_removed_node)

                for class_need_to_be_removed_node in class_need_to_be_removed_nodes:  # noqa: E501
                    node.body.remove(class_need_to_be_removed_node)

    # lazy add new ast.ImportFrom node to ast.Module
    for insert_idx, temp_node in insert_idx_and_node.items():
        ast_tree.body.insert(insert_idx, temp_node)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(format_code(ast.unparse(ast_tree)))


def _replace_config_scope_to_pack(file_path: str):
    """Replace the config scope from "mmxxx" to "pack".

    Args:
        file_path (str): The file path.
    """

    with open(file_path) as file:
        content = file.read()

    # replace all the ``_scope_``
    target_pattern = r"_scope_='[^']+'"
    replacement_string = r"_scope_='pack'"
    updated_content = re.sub(target_pattern, replacement_string, content)

    # replace all the ``default_scope``
    target_pattern = r"default_scope = '[^']+'"
    replacement_string = r"default_scope = 'pack'"
    updated_content = re.sub(target_pattern, replacement_string,
                             updated_content)

    with open(file_path, 'w') as file:
        file.write(updated_content)


def _wrapper_all_registries_build_func(export_module_dir: str, scope: str):
    """A function to wrap all registries' build_func.

    Args:
        pack_module_dir (str): The root dir for packing modules.
        scope (str): The default scope of the config.
    """
    import importlib
    import shutil

    # copy the downstream repo.registry to pack.registry
    # and change all the registry.locations
    repo_registries = importlib.import_module('.registry', scope)
    origin_file = inspect.getfile(repo_registries)
    shutil.copy(origin_file, osp.join(export_module_dir, 'registry.py'))

    with open(
            osp.join(export_module_dir, 'registry.py'), 'r+',
            encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Constant):
                if 'mm' in node.value:
                    node.value = 'pack.' + node.value.split('.', 1)[-1]

    with open(osp.join(export_module_dir, 'registry.py'), 'w') as f:
        f.write(format_code(ast.unparse(ast_tree)))

    # prevent circular registration
    Registry._extra_module_set = set()

    # record the exported module for postprocessing the importfrom path
    Registry._module_path_dict = {}

    # prevent circular wrapper
    if Registry.build.__name__ == 'wrapper':
        Registry.build = _build(Registry.init_build_func, export_module_dir)
        Registry.get = _get(Registry.init_get_func, export_module_dir)
    else:
        Registry.init_build_func = copy.deepcopy(Registry.build)
        Registry.init_get_func = copy.deepcopy(Registry.get)
        Registry.build = _build(Registry.build, export_module_dir)
        Registry.get = _get(Registry.get, export_module_dir)


@lru_cache
def _export_module(self, obj_cls: type, pack_module_dir, obj_type: str):
    """Export module.

    This function will get the object's file and export to
    ``pack_module_dir``.

    If the object is built by ``MODELS`` registry, all the objects
    as the top classes in this file, will be iteratively flattened.
    Else will be directly exported.

    The flatten logic is:
        1. get the origin file of object, which built
            by ``MODELS.build()``
        2. get all the classes in the origin file
        3. flatten all the classes but not only the object
        4. call ``flatten_module()`` to finish flatten
            according to ``class.mro()``

    Args:
        obj (object): The object to be flatten.
    """
    # find the file by obj class
    file_path = inspect.getfile(obj_cls)

    if osp.exists(file_path):
        print_log(
            f'[package: {pack_module_dir} ] building class: '
            f'{obj_cls.__name__} from file: {file_path}.',
            logger='current',
            level=logging.INFO)
    else:
        raise FileExistsError(f"file [{file_path}] doesn't exist.")

    # local origin module
    module = obj_cls.__module__
    parent = module.split('.')[0]
    new_module = module.replace(parent, 'pack')

    # Not necessary to export module implemented in `mmcv` and `mmengine`
    if parent not in set(OFFICIAL_MODULES) - {'mmcv', 'mmengine'}:

        with open(file_path, encoding='utf-8') as f:
            top_ast_tree = ast.parse(f.read())

        # deal with relative import
        ImportResolverTransformer(module).visit(top_ast_tree)

        # NOTE: ``MODELS.build()`` means to flatten model module
        if self.name == 'model':

            # record all the class needed to be flattened
            need_to_be_flattened_class_names = []
            for node in top_ast_tree.body:
                if isinstance(node, ast.ClassDef):
                    need_to_be_flattened_class_names.append(node.name)

            imported_module = importlib.import_module(obj_cls.__module__)
            for cls_name in need_to_be_flattened_class_names:

                # record the exported module for postprocessing the importfrom path  # noqa: E501
                self._module_path_dict[cls_name] = new_module

                cls = getattr(imported_module, cls_name)

                for super_cls in cls.__bases__:

                    # the class only will be flattened when:
                    #   1. super class doesn't exist in this file
                    #   2. and super class is not base class
                    #   3. and super class is not torch module
                    if super_cls.__name__\
                        not in need_to_be_flattened_class_names \
                        and (super_cls not in [BaseModule,
                                               BaseModel,
                                               BaseDataPreprocessor,
                                               ImgDataPreprocessor]) \
                            and 'torch' not in super_cls.__module__:  # noqa: E501

                        print(f'need_flatten: {cls_name} super {super_cls}')
                        flatten_module(top_ast_tree, cls)
                        break
            postprocess_super(top_ast_tree)


        else:
            self._module_path_dict[
                obj_cls.__name__] = new_module

        # add ``register_module(force=True)`` to cover the registered modules  # noqa: E501
        RegisterModuleTransformer().visit(top_ast_tree)

        # unparse ast tree and save reformat code
        new_file_path = new_module.strip('pack.').replace('.', '/') + '.py'
        new_file_path = osp.join(pack_module_dir, new_file_path)
        new_dir = osp.dirname(new_file_path)
        mkdir_or_exist(new_dir)

        with open(new_file_path, mode='w') as f:
            f.write(format_code(ast.unparse(top_ast_tree)))

    # Downstream repo could register torch module into Registry, such as
    # registering `torch.nn.Linear` into `MODELS`. We need to reserve these
    # codes in the exported module.
    elif 'torch' in module.split('.')[0]:

        # get the root registry, because it can get all the modules
        # had been registered.
        root_registry = self if self.parent is None else self.parent
        if (obj_type not in self._extra_module_set) and (
                root_registry.init_get_func(obj_type) is
                None):  # TODO 这里不应该是 obj_cls.name因为注册名字可能不一样
            self._extra_module_set.add(obj_type)
            with open(osp.join(pack_module_dir, 'registry.py'), 'a') as f:

                # TODO: the downstream repo registries' name maybe
                # different with mmengine for example: EVALUATOR in
                # mmengine, EVALUATORS in mmdet.
                f.write('\n')
                f.write(f'from {module} import {obj_cls.__name__}\n')
                f.write(
                    f"{REGISTRY_TYPE[self.name]}.register_module('{obj_type}', module={obj_cls.__name__}, force=True)"  # noqa: E501
                )


def _build(build_func: Callable, pack_module_dir: str):
    """wrap Registry.build()

    Args:
        build_func (Callable): ``Registry.build()``, which will be wrapped.
        pack_module_dir (str): Modules export path.
    """

    def wrapper(self, cfg: dict, *args, **kwargs):

        # obj is class instanace
        obj = build_func(self, cfg, *args, **kwargs)
        args = cfg.copy()  # type: ignore
        obj_type = args.pop('type')  # type: ignore
        obj_type = obj_type if isinstance(obj_type, str) else obj_type.__name__

        # modules in ``torch.nn.Sequential`` should be respectively exported
        if isinstance(obj, torch.nn.Sequential):
            for children in obj.children():
                _export_module(self, children.__class__, pack_module_dir,
                               obj_type)
        else:
            _export_module(self, obj.__class__, pack_module_dir, obj_type)

        return obj

    return wrapper


def _get(get_func: Callable, pack_module_dir: str):
    """wrap Registry.get()

    Args:
        get_func (Callable): ``Registry.get()``, which will be wrapped.
        pack_module_dir (str): Modules export path.
    """

    def wrapper(self, key: str):

        obj_cls = get_func(self, key)

        _export_module(self, obj_cls, pack_module_dir, obj_type=key)

        return obj_cls

    return wrapper


def flatten_module(top_ast_tree: ast.Module, obj_cls: type):
    """Flatten the module. (Key Interface)

    The logic of the ``flatten_module`` are as below.
    First, get the inheritance_chain by ``class.mro()`` and prune it.
    Second, get the file of chosen top class and parse it to
        be ``top_ast_tree``.
    Third, call ``init_prepare()`` to collect the information of
        ``top_ast_tree``.

    Last, for each super class in the inheritance_chain, we will do:
        1. parse the super class file as  ``super_ast_tree`` and
            do preprocess.
        2. call ``flatten_model()`` to visit necessary node
            in ``super_ast_tree`` to change needed flattened class node and
            record the information for flatten.
        3. call ``postprocess_ast_tree()`` with the information got from
           ``flatten_model()`` to change the ``top_ast_tree``.

    In summary, ``top_ast_tree`` is the most important ast tree maintained and
    updated from the begin to the end.

    Args:
        top_ast_tree (ast.Module): The top ast tree contains the classes
            directly called, which is continually updated.
        obj_cls (object): The chosen top class to be flattened.
    """
    print(
        f'------------- flatten model [{obj_cls.__name__}] -------------\n', )
    print(f'*[mro]: {obj_cls.mro()}\n')

    # get inheritance_chain
    inheritance_chain = []
    for cls in obj_cls.mro():
        if cls in [
                BaseModule, BaseModel, BaseDataPreprocessor,
                ImgDataPreprocessor, obj_cls
        ] or 'torch' in cls.__module__:
            break
        inheritance_chain.append(cls)
    print(f'*[inheritance_chain]: {inheritance_chain}\n')

    # collect the init information of ``top_ast_tree``
    import_from_dict_top, import_list_top, class_dict_top, assign_list_top, \
        try_list_top, if_list_top, import_from_asname_dict_top \
        = init_prepare(top_ast_tree, obj_cls.__name__)

    # iteratively deal with the super class
    for cls in inheritance_chain:

        modul_pth = inspect.getfile(cls)
        with open(modul_pth) as f:
            super_ast_tree = ast.parse(f.read())

        ImportResolverTransformer(cls.__module__).visit(super_ast_tree)

        # collect the difference between ``top_ast_tree`` and ``super_ast_tree``  # noqa: E501
        used_module_dict_super, extra_import_list_super, \
            extra_import_from_dict_super = flatten_model(
                super_ast_tree=super_ast_tree,
                class_dict_top=class_dict_top,
                importfrom_dict_top=import_from_dict_top,
                import_list_top=import_list_top,
                assign_list_top=assign_list_top,
                if_list_top=if_list_top,
                try_list_top=try_list_top,
                importfrom_asname_dict_top=import_from_asname_dict_top)

        # update ``top_ast_tree``
        postprocess_top_ast_tree(
            super_ast_tree,
            top_ast_tree,
            used_module_dict_super,
            extra_import_from_dict_super,
            extra_import_list_super,
            class_dict_top,
            assign_list_top,
            try_list_top,
            if_list_top,
            import_from_dict_top,
            import_list_top,
        )

    print(
        f'------------- flatten model [{obj_cls.__name__}] -------------\n', )


class RegisterModuleTransformer(ast.NodeTransformer):
    """Deal with repeatedly registering same module.

    Add "force=True" to register_module(force=True) for covering registered
    modules.
    """

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.attr == 'register_module':
                    new_keyword = ast.keyword(
                        arg='force', value=ast.NameConstant(value=True))
                    if node.keywords is None:
                        node.keywords = [new_keyword]
                    else:
                        node.keywords.append(new_keyword)
        return node


class ImportResolverTransformer(ast.NodeTransformer):
    """Deal with the relative import problem.

    Args:
        import_prefix (str): The import prefix for the visit ast code

    Examples:
        >>> # file_path = '/home/username/miniconda3/envs/env_name/lib' \
        >>>               '/python3.9/site-packages/mmdet/models/detectors' \
        >>>               '/dino.py'
        >>> import_prefix = mmdet.models.detector
    """

    def __init__(self, import_prefix: str):
        super().__init__()
        self.import_prefix = import_prefix

    def visit_ImportFrom(self, node):
        matched = self._match_alias_registry(node)
        if matched is not None:
            # In an ideal scenario, the `ImportResolverTransformer` would modify
            # the import sources of all `Registry` from downstream algorithm
            # libraries (`mmdet`) to `pack`, for example, convert
            # `from mmdet.models import DETECTORS` to
            # `from pack.models import DETECTORS`.

            # However, some algorithm libraries, such as `mmpose`, provide aliases
            # for `MODELS`, `TASK_UTILS`, and other registries,
            # as seen here: https://github.com/open-mmlab/mmpose/blob/537bd8e543ab463fb55120d5caaa1ae22d6aaf06/mmpose/models/builder.py#L13.
            
            # For these registries with aliases, we cannot directly import from
            # `pack.registry` because `pack.registry` is copied from
            # `mmpose.registry` and does not contain these aliases.

            # Therefore, we gather all registries with aliases under
            # `mim.utils.mmpack.patch_utils` and hardcode the redirection
            # of import sources.
            if matched == 'MODELS':
                node.module = 'mim.utils.mmpack.patch_utils.patch_model'
            elif matched == 'TASK_UTILS':
                node.module = 'mim.utils.mmpack.patch_utils.patch_task'
            node.level = 0
            return node
        
        if node.level == 0:
            if 'registry' in node.module \
                    and not node.module.startswith('mmengine'):
                node.module = 'pack.registry'

        # deal with relative import
        else:
            import_prefix = '.'.join(
                self.import_prefix.split('.')[:-node.level])
            if node.module is not None:
                node.module = import_prefix + '.' + node.module
            else:
                # from . import xxx
                node.module = import_prefix
            node.level = 0
        return node

    # TODO: resolve Import Node

    def _match_alias_registry(self, node) -> Optional[str]:
        match_patch_key = None
        for key, list_value in BUILDER_TRANS.items():
            for alias in node.names:
                if alias.name in list_value:
                    match_patch_key = key
                    break

            if match_patch_key is not None:
                break
        return match_patch_key
