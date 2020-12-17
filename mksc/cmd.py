import sys
import os
import shutil

__version__ = '1.1.16'

def make_workspace(name):
    """
    创建项目的工作目录与脚本文件

    Args:
        name: 项目名
    """
    name = f'{name}_{__version__}'
    if os.path.exists(name):
        raise TypeError(f"Folder [{name}] is exists already, Please check out the work path")
    else:
        templates = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'template')
        project = os.path.join(os.getcwd(), name)
        shutil.copytree(templates, project)

    os.mkdir(os.path.join(os.getcwd(), name, 'result'))
    os.mkdir(os.path.join(os.getcwd(), name, 'data'))

def main():
    """
    命令行工具程序主入口
    """
    if len(sys.argv) == 1:
        return "CMD FORMAT: \n\tmksc project_name1 project_name2 ...\nPlease delivery one argument at least"
    else:
        for project_name in sys.argv[1:]:
            make_workspace(project_name)


if __name__ == "__main__":
    main()
