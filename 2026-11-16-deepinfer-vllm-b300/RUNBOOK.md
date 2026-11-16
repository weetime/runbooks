# B300 现场执行手册（仅交付流程，本轮不执行构建）

入口：`bash scripts/b300.sh <阶段>`。默认只打印plan，不自动构建或发压。
遵循仓库`BENCHMARK_POLICY.md`；issue为runbooks#1。

## 一、联网准备机：准备镜像与数据

这一步应由后续操作者执行，本轮**没有构建或发布DeepInfer build-kit**。
准备机需要Linux x86_64、Git、Python3，以及Docker build；用nerdctl build时还需已配置BuildKit。
不需要GPU。源代码和编译依赖在这里下载，不能留到离线B300再下载。

```bash
# 只在后续明确执行准备时运行。此命令会下载依赖并构建build-kit。
RUNTIME=docker BUILD_CONTEXT=/data/new-deepinfer-context \
  BUILD_KIT_TAG=swr.cn-north-4.myhuaweicloud.com/runner/deepinfer-buildkit:a679bb86-cu132 \
  bash scripts/b300.sh prepare
# 操作者使用既有登录推送，随后核验manifest digest；凭据不要写到配置或命令行。
docker push swr.cn-north-4.myhuaweicloud.com/runner/deepinfer-buildkit:a679bb86-cu132
```

build-kit来自固定CUDA13.2基础镜像，装好Python3.12、GCC13、Torch2.13+cu132、NCCL2.32.3及构建依赖，
带齐固定DeepInfer和递归submodule源码，但**不编译DeepInfer和四个vendor库**。
不要把build-kit称作已验证推理镜像。

vLLM与AIPerf的精确源digest见`manifests/images.json`。后续联网同步命令：

```bash
python3 scripts/mirror_images.py manifests/images.json             # 先预览
python3 scripts/mirror_images.py manifests/images.json --execute   # 明确执行时才同步
```

当前已核验的SWR制品为AIPerf、CUDA基础镜像和32MB文本数据镜像；vLLM大镜像同步未完成。
完整镜像列表和未就绪项均以清单中的status为准，不使用latest或猜测的digest。
完全断网现场需提前导出镜像tar、生成SHA256SUMS，拷到共享盘后先校验再load；有SWR白名单时按digest pull。

## 二、B300：配置与制品检查

```bash
cp configs/b300.env.example configs/b300.env
# 填现场GPU_IDS、模型/共享盘路径、刚核验的BUILD_KIT_IMAGE@sha256和其他镜像digest。
export CONFIG="$PWD/configs/b300.env"
bash scripts/b300.sh plan
bash scripts/b300.sh pull       # 或 ARCHIVE_DIR=/data/image-archives bash scripts/b300.sh load
bash scripts/b300.sh data       # 已有同SHA数据则复用，不覆盖
bash scripts/b300.sh check
```

需要Linux x86_64和已配置NVIDIA容器支持的Docker/containerd；脚本不会安装驱动或启动宿主机服务。
B300/SM103、驱动与CUDA13.2的兼容性、空闲GPU、互联和CPU内存必须现场核对。
`GPU_IDS`必须显式填写，不默认all；检查到已有计算进程即拒绝。Kubernetes管理的节点还需核对调度器分配，
不能只以nvidia-smi空闲认定已获得资源。

数据镜像：
`swr.cn-north-4.myhuaweicloud.com/runner/benchmark-data@sha256:983d4875ac1540e1f5c52007015d0e79203690ad345954e9b36aa1f8207a3aeb`。
包含独立身份的14文件源子集、7×128正式请求、7×3预热请求、对应HTTP候选和model metadata。
原生与HTTP文件合计1,834条准备记录。源子集SHA256SUMS为
`67a8678f69b298b9dd8ce7a28d679b1bb23e084a0437c022bc3767fa416273d8`。
它不是完整standard-v1，不含完整Agent/CC或权重；完整约600MB归档只在本地保留并校验。
HTTP候选哈希一致不代表服务端编码相同，仍需实际请求/token对账。

## 三、B300：先构建DeepInfer镜像

```bash
bash scripts/b300.sh build
```

该阶段在GPU范围明确的B300容器执行，`--network none`，使用已装依赖和完整源码。
仅调用固定官方源码的`build_vendor.sh`和`scripts/build.sh build`，不在线pip、Git或下载模型。
构建日志、源码SHA、wheel SHA、依赖版本和设备信息留在`WORK_DIR/build-<时间>/`。
失败不会commit成成功镜像；保留容器与完整日志排查，不静默改参数或fallback。
构建成功仅说明编译完成，不代表模型正确、速度达标或通信通过。

## 四、模型和功能门禁

模型必须是同一原厂`DeepSeek-V4.1-Flash-1017` mixed MXFP4/MXFP8快照，revision
`08c8657f02291e11b6f6b63e0e16014f8e748518`。权重、tokenizer、encoding和模型代码只读挂载。
不以W8A8、0909、第三方量化或目录存在代替本检查点。

```bash
MODEL_MANIFEST=/data/evidence/model-full.json bash scripts/b300.sh freeze-model
MODEL_MANIFEST=/data/evidence/model-full.json bash scripts/b300.sh smoke
# 编译镜像标candidate后发布并记录远端digest，写回配置DI_IMAGE。
DI_SWR_TAG=swr.cn-north-4.myhuaweicloud.com/runner/deepinfer:a679bb86-b300-candidate \
  bash scripts/b300.sh publish
# 完全隔离时先导出，另由联网中转机发布/核验。
EXPORT_ARCHIVE=/data/artifacts/deepinfer-b300-candidate.tar bash scripts/b300.sh export
```

smoke是官方离线文本/视觉/结构化样例，EP8/CP8/DP1需8卡；它不是完整G2，也不是性能结果。
执行模型请求前先冻结最终配方和资源；候选CP8、窗口、batch等实质偏离按测试纪律确认。
原厂无Jinja模板，必须使用真实encoding/tokenizer；NCCL官方覆盖差异要保留，不能称pip-check全通过。

## 五、逐臂建门禁记录、单流，再发压

下面是现场示例；每个臂和场景使用自己的记录。模板初始所有门禁/批准值均为false。

```bash
export ENGINE=deepinfer
export RECIPE_FILE="$PWD/configs/deepinfer-candidate.json"
export GATES_FILE=/data/evidence/deepinfer-s8k-gates.json
bash scripts/b300.sh gates
# 先审查制品、完整模型哈希及资源/配方授权，记录真实G0证据。
POINT_ID=deepinfer-recipe-probe bash scripts/b300.sh probe
# 核对本配方实际启动日志、KV/图/DSpark/并行及短答/工具/视觉/长检索，再记录真实G1/G2。
POINT_ID=deepinfer-g3-start1 bash scripts/b300.sh g3
# 复核g3.json、原始输出、token对账和停止线，才可在记录中写G3=true。
POINT_ID=deepinfer-s8k-b8-start1 SCENARIO=s8k BATCH_SIZE=8 REQUEST_COUNT=128 \
  bash scripts/b300.sh bench
```

vLLM改为`ENGINE=vllm`、相应candidate JSON、独立门禁和point ID。
probe只用于启动与文本观察，不自动通过G1/G2；G3为同次起服24个不同8K/512请求，按8条分3组。
G3的native输出chunk计时与HTTP时延分开，不能把chunk数当token数；计时接口不适配或token对账不符即失败。
G3下界、3组极差/中位数和已登记本机参照按冻结判据审核，小样本不作可靠P95或容量结论。

普通bench一次命令一次起服、一个batch点，输出仍为partial，需人工质量/计数器/最终参数核验。
它不是自动完成全部标准矩阵的一键campaign；正式实验应登记启动次数与缓存状态，按纪律组织关键点A→B→A→B→A→B，
不得把各命令谎称为同一次起服。完整标准档的实例复用/调度和GPU适配仍需现场验证。
批大小B不等于闭环并发C；主表实际生成tok/s与官方全接受率sweep分表。

## 六、结果和未支持场景

始终保留完整server/client日志、逐请求输出、point.json、指标前后差和负载中整机1Hz快照；结果只追加不覆盖。
目前没有B300硬件，脚本仅做静态/CPU检查，不宣称实机可用或已验收。
DeepInfer公开版没有完整HTTP服务；相关AIPerf、Agent/CC、泊松容量项在适配就绪前记not_run。
原生客户端运行在引擎进程内，不能与独立Pod的AIPerf数据混表，也不以其数值冒充在线服务成绩。
完成现场验证后回填issue #41。未通过G2/G3或停止线触发时，保留证据停止，不擅自关优化后继续沿用配方名。
