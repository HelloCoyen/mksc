import sys
import os
import shutil

def make_workspace(name):
    """
    创建项目的工作目录与脚本文件

    Args:
        name: 项目名
    """

    if os.path.exists(name):
        raise TypeError(f"Folder [{name}] is exists already, Please check out the work path")
    else:
        templates = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'template')
        project = os.path.join(os.getcwd(), name)
        shutil.copytree(templates, project)

    os.mkdir(os.path.join(os.getcwd(), name, 'result'))
    os.mkdir(os.path.join(os.getcwd(), name, 'data'))

    with open(os.path.join(project, 'config', 'configuration.ini'), "r", encoding='utf-8') as f:
        content = f.read()
    with open(os.path.join(project, 'config', 'configuration.ini'), "w", encoding='utf-8') as f:
        content = content % (os.path.join(os.getcwd(), name))
        f.write(content)

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
