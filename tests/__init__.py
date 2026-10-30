# -*- coding: utf-8 -*-
"""测试包：隔离数据库与模型配置，把仓库根目录加进 sys.path。

两件事必须在这里做，而且要在 import app 之前：

1. `DB_URL`：`app.core.store` 在 import 时就建库，测试不能污染 data/questedu.db
2. `llm.ENV_PATH`：指向一个不存在的临时文件。否则跑测试时会读到开发者自己的
   .env——比如本机正开着 LM Studio 并且模式是 local，测试就会真的去连
   127.0.0.1:1234 然后卡在那里等模型思考。测试必须能离线、可重复地跑。
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TEST_DB = Path(tempfile.gettempdir()) / "questedu_test.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DB_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
os.environ.pop("STORE", None)

# 清掉可能从外部带进来的模型配置，再把 .env 指到空文件
for _k in [k for k in os.environ if k.startswith(("LLM_", "INFERENCE_MODE"))]:
    os.environ.pop(_k, None)
from app.core import llm  # noqa: E402 必须在环境变量清理之后

llm.ENV_PATH = Path(tempfile.gettempdir()) / "questedu_no_such.env"
if llm.ENV_PATH.exists():
    llm.ENV_PATH.unlink()
llm.reload()
assert llm.mode() == "mock", f"测试环境没隔离干净，当前模式是 {llm.mode()}"

# 访问码配置同理：不能写进仓库根目录的 .access.json
from app.core import access as _access  # noqa: E402

_access.CONFIG_PATH = Path(tempfile.gettempdir()) / "questedu_access.json"
if _access.CONFIG_PATH.exists():
    _access.CONFIG_PATH.unlink()
