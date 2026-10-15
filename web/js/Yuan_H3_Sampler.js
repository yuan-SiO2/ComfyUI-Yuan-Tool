/** Yuan Tool · H3 渐进式采样器前端：把「高分辨率模型」输入端口排到「低分辨率模型」正下方，实现双模型接入的相邻布局。 */
const { app } = window.comfyAPI.app;

const SAMPLERS = new Set(["Yuan_H3ProgressiveSampler"]);

function orderModelInputs(node) {
    const inputs = node.inputs;
    if (!inputs) return;
    const low = inputs.findIndex((input) => input.name === "low_res_model");
    const high = inputs.findIndex((input) => input.name === "high_res_model");
    if (low < 0 || high < 0 || high === low + 1) return;
    const [input] = inputs.splice(high, 1);
    inputs.splice(inputs.findIndex((input) => input.name === "low_res_model") + 1, 0, input);
    // 连线按槽位索引寻址，重排后需把已保存连线的目标槽位同步到同一输入
    for (let slot = 0; slot < inputs.length; slot++) {
        const linkId = inputs[slot].link;
        if (linkId == null) continue;
        const link = node.graph?.links[linkId];
        if (link) link.target_slot = slot;
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Comfy.YuanTool.H3ProgressiveSamplerInputOrder",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!SAMPLERS.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            orderModelInputs(this);
            return result;
        };
    },
    afterConfigureGraph() {
        // 等待图反序列化恢复所有节点与连线后再重排
        for (const node of app.graph?._nodes ?? []) {
            if (SAMPLERS.has(node.type)) orderModelInputs(node);
        }
    },
});
