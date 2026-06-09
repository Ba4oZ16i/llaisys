#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/model.hpp"
#include <cstdint>

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *model;
        LlaisysQwen2Weights weights;
    };
    __export LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        LlaisysQwen2Model *qwen2model = new LlaisysQwen2Model;
        llaisys::models::Qwen2Model *model = new llaisys::models::Qwen2Model(*meta, device, device_ids, ndevice);
        qwen2model->model = model;
        qwen2model->model->fillWeights(&(qwen2model->weights));
        return qwen2model;
    }
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model->model;
        delete model;
    }

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return &(model->weights);
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temperature, int64_t top_k, float top_p) {
        return model->model->infer(token_ids, ntoken, temperature, top_k, top_p);
    }
}