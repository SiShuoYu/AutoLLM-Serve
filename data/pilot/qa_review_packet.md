# CiteTune-CN QA 人工审核包

> 这份文件用于人工核对，不是已批准的数据集。请同时检查问题、答案和原文证据。

## train-answerable-0001

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 对于 Kubernetes v1.23 及更早版本，如何根据 kubelet 命令行参数判断节点是在使用 dockershim 还是某个远程容器运行时？

**参考答案：** 如果 `--container-runtime` 和 `--container-runtime-endpoint` 都不存在，或 `--container-runtime` 的值不是 `remote`，则节点通过 dockershim 套接字使用 Docker Engine。如果设置了 `--container-runtime-endpoint`，应查看套接字名称判断运行时，例如 `unix:///run/containerd/containerd.sock` 表示 containerd。

**引用 ID：** `9d18e3c6ce13fdf5-0004`

**来源：** [content/zh-cn/docs/tasks/administer-cluster/migrating-from-dockershim/find-out-runtime-you-use.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/tasks/administer-cluster/migrating-from-dockershim/find-out-runtime-you-use.md)

**原文证据：**

    *   If your nodes use Kubernetes v1.23 and earlier and these flags aren't
            present or if the `--container-runtime` flag is not `remote`,
            you use the dockershim socket with Docker Engine. The `--container-runtime` command line
            argument is not available in Kubernetes v1.27 and later.
        *   If the `--container-runtime-endpoint` flag is present, check the socket
            name to find out which runtime you use. For example,
            `unix:///run/containerd/containerd.sock` is the containerd endpoint.
    -->
    2. 在命令的输出中，查找 `--container-runtime` 和 `--container-runtime-endpoint` 标志。

    * 如果你的节点使用 Kubernetes v1.23 或更早的版本，这两个参数不存在，
         或者 `--container-runtime` 标志值不是 `remote`，则你在通过 dockershim 套接字使用
         Docker Engine。
         在 Kubernetes v1.27 及以后的版本中，`--container-runtime` 命令行参数不再可用。
       * 如果设置了 `--container-runtime-endpoint` 参数，查看套接字名称即可得知当前使用的运行时。
         如若套接字 `unix:///run/containerd/containerd.sock` 是 containerd 的端点。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0002

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`rejected`

**拒绝原因：** 文本块以“此标签”开头，但没有给出标签名称，脱离上一块后上下文不完整。

---

## train-answerable-0003

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** NodeRestriction 准入插件为什么通常禁止 kubelet 使用其他以 `kubernetes.io` 或 `k8s.io` 开头的标签？

**参考答案：** 这些前缀下的其他标签属于保留范围，通常禁止 kubelet 使用是为了防止未经授权的自我标记。

**引用 ID：** `6f2000effc001cfe-0043`

**来源：** [content/zh-cn/docs/reference/access-authn-authz/admission-controllers.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/access-authn-authz/admission-controllers.md)

**原文证据：**

    <!--
    Use of any other labels under the `kubernetes.io` or `k8s.io` prefixes by kubelets is reserved,
    and may be disallowed or allowed by the `NodeRestriction` admission plugin in the future.
    * **Reserved**:
      Use of any other labels under the `kubernetes.io` or `k8s.io` prefixes by kubelets is reserved.
      The `NodeRestriction` admission plugin generally disallows these to prevent unauthorized self-labeling,
      but may allow additional labels under these prefixes in the future as part of future features.
    -->
    **保留**：
      以 `kubernetes.io` 或 `k8s.io` 为前缀的所有其他标签都限制 kubelet 使用。
      `NodeRestriction` 准入插件通常禁止这些操作，以防止未经授权的自我标记，
      但未来可能会作为新特性的一部分允许在这些前缀下添加额外的标签。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0004

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** Kubernetes 的 NetworkPolicy API 可以控制哪些对象之间的网络流量？

**参考答案：** 它可以控制 Pod 之间的流量，以及 Pod 与集群外部世界之间的流量。

**引用 ID：** `469313127ecdf616-0003`

**来源：** [content/zh-cn/docs/concepts/services-networking/_index.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/concepts/services-networking/_index.md)

**原文证据：**

    * [NetworkPolicy](/docs/concepts/services-networking/network-policies) is a built-in
      Kubernetes API that allows you to control traffic between pods, or between pods and
      the outside world.
    -->
    * [Gateway](/zh-cn/docs/concepts/services-networking/gateway/) API
      （或其前身 [Ingress](/zh-cn/docs/concepts/services-networking/ingress/)）
      使得集群外部的客户端能够访问 Service。

    * 当使用受支持的 {{< glossary_tooltip term_id="cloud-provider">}} 时，通过 Service API 的
        [`type: LoadBalancer`](/zh-cn/docs/concepts/services-networking/service/#loadbalancer)
        可以使用一种更简单但可配置性较低的集群 Ingress 机制。

    * [NetworkPolicy](/zh-cn/docs/concepts/services-networking/network-policies)
      是一个内置的 Kubernetes API，允许你控制 Pod 之间的流量或 Pod 与外部世界之间的流量。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0005

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 运行 `update-imported-docs.py` 生成 Kubernetes 参考文档时，需要提供哪两个参数？

**参考答案：** 需要提供一个 YAML 配置文件（`reference.yml`）和一个发行版本字符串，例如 `1.17`。

**引用 ID：** `3fb5a8f018d9669c-0003`

**来源：** [content/zh-cn/docs/contribute/generate-ref-docs/quickstart.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/contribute/generate-ref-docs/quickstart.md)

**原文证据：**

    * `K8S_RELEASE`
    * `K8S_ROOT`
    * `K8S_WEBROOT`
    -->
    脚本 `update-imported-docs.py` 基于 Kubernetes 源代码生成参考文档。
    过程中会在你的机器的 `/tmp` 目录下创建临时目录，克隆所需要的仓库
    `kubernetes/kubernetes` 和 `kubernetes-sigs/reference-docs` 到此临时目录。
    脚本会将 `GOPATH` 环境变量设置为指向此临时目录。
    此外，脚本会设置三个环境变量：

    * `K8S_RELEASE`
    * `K8S_ROOT`
    * `K8S_WEBROOT`

    <!--
    The script requires two arguments to run successfully:

    * A YAML configuration file (`reference.yml`)
    * A release version, for example:`1.17`

    The configuration file contains a `generate-command` field.
    The `generate-command` field defines a series of build instructions
    from `kubernetes-sigs/reference-docs/Makefile`. The `K8S_RELEASE` variable
    determines the version of the release.
    -->
    脚本需要两个参数才能成功运行：

    * 一个 YAML 配置文件（`reference.yml`）
    * 一个发行版本字符串，例如：`1.17`

    配置文件中包含 `generate-command` 字段，其中定义了一系列来自于
    `kubernetes-sigs/reference-docs/Makefile` 的构建指令。
    变量 `K8S_RELEASE` 用来确定所针对的发行版本。

    <!--
    The `update-imported-docs.py` script performs the following steps:

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0006

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 一个有 500 个节点的集群将 `percentageOfNodesToScore` 设为 30 时，调度器何时停止继续寻找可行节点？

**参考答案：** 调度器找到 150 个可行节点后会停止继续寻找，但无论该值如何，仍会尝试至少找到 `minFeasibleNodesToFind` 个可行节点。

**引用 ID：** `3c3f5620ff901623-0050`

**来源：** [content/zh-cn/docs/reference/config-api/kube-scheduler-config.v1.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/config-api/kube-scheduler-config.v1.md)

**原文证据：**

    l be scored. It will override global PercentageOfNodesToScore. If it is empty,
    global PercentageOfNodesToScore will be used.
      -->
       <p>percentageOfNodesToScore 是已发现可运行 Pod 的节点与所有节点的百分比，
       调度器所发现的可行节点到达此阈值时，将停止在集群中继续搜索可行节点。
    这有助于提高调度器的性能。无论此标志的值是多少，调度器总是尝试至少找到 “minFeasibleNodesToFind” 个可行的节点。
    例如：如果集群大小为 500 个节点并且此标志的值为 30，则调度器在找到 150 个可行节点后将停止寻找更多可行的节点。
    当值为 0 时，默认百分比（根据集群大小为 5% - 50%）的节点将被评分。此设置值将覆盖全局的 PercentageOfNodesToScore 值。
    如果为空，将使用全局 PercentageOfNodesToScore。</p>
    </td>
    </tr>
    <tr><td><code>plugins</code> <B><!--[Required]-->[必需]</B><br/>
    <a href="#kubescheduler-config-k8s-io-v1-Plugins"><code>Plugins</code></a>
    </td>
    <td>
       <!--
       Plugins specify the set of plugins that should be enabled or disabled.
    Enabled plugins are the ones that should be enabled in addition to the
    default plugins. Disabled plugins are any of the default plugins that
    should be disabled.
    When no enabled or disabled plugin is specified for an extension point,
    default plugins for that extension point will be used if there is any.
    If a QueueSort plugin is specified, the same QueueSort Plugin and
    PluginConfig must be specified for all profiles.
       -->
       <p><code>plugins</code> 设置一组应该被启用或禁止的插件。
       被启用的插件是指

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0007

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 如果某项 Kubernetes 功能需要文档，但文档在发行截止日期前仍未准备好，可能产生什么后果？

**参考答案：** 该功能可能会被从对应的发行里程碑中移除。

**引用 ID：** `c1c6fbba2e20ec52-0017`

**来源：** [content/zh-cn/docs/contribute/new-content/new-features.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/contribute/new-content/new-features.md)

**原文证据：**

    If your PR has not yet been merged into the `dev-{{< skew nextMinorVersion >}}`
    branch by the release deadline, work with the docs person managing the release
    to get it in by the deadline. If your feature needs documentation and the docs
    are not ready, the feature may be removed from the milestone.
    -->
    ### 所有 PR 均经过评审且合并就绪   {#all-prs-reviewd-and-ready-to-merge}

    如果你的 PR 在发行截止日期之前尚未合并到 `dev-{{< skew nextMinorVersion >}}` 分支，
    请与负责管理该发行版本的文档团队成员一起合作，在截止期限之前将其合并。
    如果功能特性需要文档，而文档并未就绪，该特性可能会被从里程碑中去除。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## train-answerable-0008

- 数据切分：`train`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** Kubernetes API 聚合层如何让定制资源使用专门的实现？

**参考答案：** 开发者可以编写并部署自己的 API 服务器；主 API 服务器会把这些定制资源的请求委托给该服务器处理，并将这些资源提供给所有客户端。

**引用 ID：** `69722cd96cfb9a1f-0015`

**来源：** [content/zh-cn/docs/concepts/extend-kubernetes/api-extension/custom-resources.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/concepts/extend-kubernetes/api-extension/custom-resources.md)

**原文证据：**

    <!--
    ## API server aggregation

    Usually, each resource in the Kubernetes API requires code that handles REST requests and manages
    persistent storage of objects. The main Kubernetes API server handles built-in resources like
    *pods* and *services*, and can also generically handle custom resources through
    [CRDs](#customresourcedefinitions).

    The [aggregation layer](/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)
    allows you to provide specialized implementations for your custom resources by writing and
    deploying your own API server.
    The main API server delegates requests to your API server for the custom resources that you handle,
    making them available to all of its clients.
    -->
    ## API 服务器聚合  {#api-server-aggregation}

    通常，Kubernetes API 中的每个资源都需要处理 REST 请求和管理对象持久性存储的代码。
    Kubernetes API 主服务器能够处理诸如 **Pod** 和 **Service** 这些内置资源，
    也可以按通用的方式通过 [CRD](#customresourcedefinitions) 来处理定制资源。

    [聚合层（Aggregation Layer）](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)
    使得你可以通过编写和部署你自己的 API 服务器来为定制资源提供特殊的实现。
    主 API 服务器将针对你要处理的定制资源的请求全部委托给你自己的 API 服务器来处理，
    同时将这些资源提供给其所有客户端。

    <!--
    ## Choosing a method for adding custom resources

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## validation-answerable-0001

- 数据切分：`validation`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** kube-scheduler 为调度队列中的 Pod 选择节点时会经过哪些主要步骤？

**参考答案：** 它先根据约束和可用资源确定合法节点，再对合法节点排序，最后把 Pod 绑定到一个合适的节点。

**引用 ID：** `3fff4adc9f291213-0000`

**来源：** [content/zh-cn/docs/reference/command-line-tools-reference/kube-scheduler.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/command-line-tools-reference/kube-scheduler.md)

**原文证据：**

    <!--
    title: kube-scheduler
    content_type: tool-reference
    weight: 30
    auto_generated: true
    description: >-

    -->

    ## {{% heading "synopsis" %}}

    <!--
    The Kubernetes scheduler is a control plane process which assigns
    Pods to Nodes. The scheduler determines which Nodes are valid placements for
    each Pod in the scheduling queue according to constraints and available
    resources. The scheduler then ranks each valid Node and binds the Pod to a
    suitable Node. Multiple different schedulers may be used within a cluster;
    kube-scheduler is the reference implementation.
    See [scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/)
    for more information about scheduling and the kube-scheduler component.
    -->
    Kubernetes 调度器是一个控制面进程，负责将 Pods 指派到节点上。
    调度器基于约束和可用资源为调度队列中每个 Pod 确定其可合法放置的节点。
    调度器之后对所有合法的节点进行排序，将 Pod 绑定到一个合适的节点。
    在同一个集群中可以使用多个不同的调度器；kube-scheduler 是其参考实现。
    参阅[调度](/zh-cn/docs/concepts/scheduling-eviction/)以获得关于调度和
    kube-scheduler 组件的更多信息。

    ```
    kube-scheduler [flags]
    ```

    ## {{% heading "options" %}}

    <table style="width: 100%; table-layout: fixed;">
    <colgroup>
    <col span="1" style="width: 10px;" />
    <col span="1" />
    </colgroup>
    <tbody>

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## validation-answerable-0002

- 数据切分：`validation`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 当 SubjectRulesReviewStatus 返回的授权规则列表不完整时，为什么仍可相信列表中已经出现的权限？

**参考答案：** 因为授权规则是累加的；即使列表不完整，只要某条规则出现在列表中，就可以确信该主体拥有相应权限。

**引用 ID：** `c59d996237fc9f5d-0001`

**来源：** [content/zh-cn/docs/reference/kubernetes-api/definitions/subject-rules-review-status-v1-authorization.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/kubernetes-api/definitions/subject-rules-review-status-v1-authorization.md)

**原文证据：**

    <!--
    SubjectRulesReviewStatus contains the result of a rules check.
    This check can be incomplete depending on the set of authorizers
    the server is configured with and any errors experienced during evaluation.
    Because authorization rules are additive, if a rule appears in a list it&#39;s
    safe to assume the subject has that permission, even if that list is incomplete.
    -->
    SubjectRulesReviewStatus 包含了规则检查的结果。
    根据服务器配置的授权器集合以及评估过程中出现的任何错误，该检查结果可能是不完整的。
    由于授权规则是累加的，因此如果某条规则出现在列表中，
    即便该列表不完整，也可以确信主体拥有相应的权限。

    <hr>

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## validation-answerable-0003

- 数据切分：`validation`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** `kubeadm upgrade apply` 的 `--allow-experimental-upgrades` 选项有什么作用？

**参考答案：** 它会把 Kubernetes 的不稳定版本显示为可选升级版本，并允许升级到 Alpha、Beta 或 RC 版本。

**引用 ID：** `a851cebdf9d35ea0-0001`

**来源：** [content/zh-cn/docs/reference/setup-tools/kubeadm/generated/kubeadm_upgrade/kubeadm_upgrade_apply.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/setup-tools/kubeadm/generated/kubeadm_upgrade/kubeadm_upgrade_apply.md)

**原文证据：**

    ```
    preflight        在升级前运行预检
    control-plane    升级控制平面
    upload-config    将 kubeadm 和 kubelet 配置上传到 ConfigMap
      /kubeadm         将 kubeadm ClusterConfiguration 上传到 ConfigMap
      /kubelet         将 kubelet 配置上传到 ConfigMap
    kubelet-config   升级此节点的 kubelet 配置
    bootstrap-token  配置启动引导令牌和 cluster-info RBAC 规则
    addon            升级默认的 kubeadm 插件
      /coredns         升级 CoreDNS 插件
      /kube-proxy      升级 kube-proxy 插件
    post-upgrade     运行升级后的任务
    ```

    <!--
    ### Options
    -->
    ### 选项

    <table style="width: 100%; table-layout: fixed;">
    <colgroup>
    <col span="1" style="width: 10px;" />
    <col span="1" />
    </colgroup>
    <tbody>

    <tr>
    <td colspan="2">--allow-experimental-upgrades</td>
    </tr>
    <tr>
    <td></td>
    <td style="line-height: 130%; word-wrap: break-word;">
    <p>
    <!--
    Show unstable versions of Kubernetes as an upgrade alternative and allow upgrading to an alpha/beta/release candidate versions of Kubernetes.
    -->
    显示 Kubernetes 的不稳定版本作为升级替代方案，并允许升级到 Kubernetes
    的 Alpha、Beta 或 RC 版本。
    </p>
    </td>
    </tr>

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## test-answerable-0001

- 数据切分：`test`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** Kubernetes 的 ABAC 策略可以依据哪些属性来匹配访问请求？

**参考答案：** 策略可以匹配主体（用户或组）、资源属性、非资源属性（例如 `/version` 或 `/apis`）以及只读属性。

**引用 ID：** `c9b2215ac7c34976-0023`

**来源：** [content/zh-cn/docs/setup/production-environment/_index.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/setup/production-environment/_index.md)

**原文证据：**

     [Examples](/docs/reference/access-authn-authz/abac/#examples) for details.
      -->
      - **基于属性的访问控制**（[ABAC](/zh-cn/docs/reference/access-authn-authz/abac/)）：
        让你能够基于集群中资源的属性来创建访问控制策略，基于对应的属性来决定允许还是拒绝访问。
        策略文件的每一行都给出版本属性（apiVersion 和 kind）以及一个规约属性的映射，
        用来匹配主体（用户或组）、资源属性、非资源属性（/version 或 /apis）和只读属性。
        参阅[示例](/zh-cn/docs/reference/access-authn-authz/abac/#examples)以了解细节。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## test-answerable-0002

- 数据切分：`test`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 在什么情况下启用 `disable-compression` 可能加快 Kubernetes 客户端请求？

**参考答案：** 当客户端与服务器之间的网络带宽充足时，取消响应压缩可省去服务器端压缩和客户端解压时间，从而加快请求，尤其是列表请求。

**引用 ID：** `6bcbce80cde2865f-0003`

**来源：** [content/zh-cn/docs/reference/config-api/client-authentication.v1beta1.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/config-api/client-authentication.v1beta1.md)

**原文证据：**

    <tr>
    <td><code>certificate-authority-data</code><br/>
    <code>[]byte</code>
    </td>
    <td>
    <!--
       CAData contains PEM-encoded certificate authority certificates.
    If empty, system roots should be used.
    -->
    此字段包含 PEM 编码的证书机构（CA）证书。
    如果为空，则使用系统的根证书。
    </td>
    </tr>

    <tr>
    <td><code>proxy-url</code><br/>
    <code>string</code>
    </td>
    <td>
    <!--
    ProxyURL is the URL to the proxy to be used for all requests to this cluster.
    -->
    此字段用来设置向集群发送所有请求时要使用的代理服务器。
    </td>
    </tr>

    <tr>
    <td><code>disable-compression</code><br/>
    <code>bool</code>
    </td>
    <td>
    <p>
    <!--
    DisableCompression allows client to opt-out of response compression for all requests to the server. This is useful
    to speed up requests (specifically lists) when client-server network bandwidth is ample, by saving time on
    compression (server-side) and decompression (client-side): https://github.com/kubernetes/kubernetes/issues/112296.
    -->
    disable-compression 允许客户端针对到服务器的所有请求选择取消响应压缩。
    当客户端服务器网络带宽充足时，这有助于通过节省压缩（服务器端）和解压缩
    （客户端）时间来加快请求（特别是列表）的速度：
    https://github.com/kubernetes/kubernetes/issues/112296。
    </p>
    </td>
    </tr>

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---

## test-answerable-0003

- 数据切分：`test`
- 队列角色：`primary`
- 当前状态：`needs_revision`

**问题：** 使用 kubectl 的 `-o wide` 输出 Pod 时，会比普通纯文本输出多包含什么信息？

**参考答案：** `-o wide` 会输出所有附加信息；对于 Pod，其中包括节点名称。

**引用 ID：** `5ac34004cbca99ca-0027`

**来源：** [content/zh-cn/docs/reference/kubectl/_index.md](https://github.com/kubernetes/website/blob/56464fe6c846523da555b49fbb012f0e270871ae/content/zh-cn/docs/reference/kubectl/_index.md)

**原文证据：**

    。
    `-o custom-columns-file=<filename>` | 使用 `<filename>` 文件中的[自定义列](#custom-columns)模板打印表。
    `-o json`                           | 输出 JSON 格式的 API 对象。
    `-o jsonpath=<template>`            | 打印 [jsonpath](/zh-cn/docs/reference/kubectl/jsonpath/) 表达式定义的字段。
    `-o jsonpath-file=<filename>`       | 打印 `<filename>` 文件中 [jsonpath](/zh-cn/docs/reference/kubectl/jsonpath/) 表达式定义的字段。
    `-o kyaml`                          | 输出 [KYAML](/zh-cn/docs/reference/encodings/kyaml/) 格式的 API 对象（Beta）。
    `-o name`                           | 仅打印资源名称而不打印任何其他内容。
    `-o wide`                           | 以纯文本格式输出，包含所有附加信息。对于 Pod 包含节点名。
    `-o yaml`                           | 输出 YAML 格式的 API 对象。KYAML 是 YAML 的一种实验性的 Kubernetes 专用方言，可以像 YAML 一样进行解析。

- [ ] 批准：问题自然，答案完全由证据支持
- [ ] 需修改：在下面写明问题
- [ ] 拒绝：来源本身不适合出题
- 审核备注：
- 审核人：

---
