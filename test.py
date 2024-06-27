    # def _register_accum_grad_hook(self):
    #     import torch.distributed._functional_collectives as fcol

    #     def compiled_accum_grad_hook(
    #         param,
    #         *,
    #         param_index: int,
    #     ):
    #         if not self.require_backward_grad_sync:
    #             return

    #         if param.grad is None:
    #             return

    #         if self._comm_hooks:
    #             for hook, state in self._comm_hooks:
    #                 hook(state, (param.grad, param))
    #         else:
    #             gradient = param.grad / self.process_group.size()
    #             gradient = fcol.all_reduce(gradient, "sum", self.process_group)
    #             param.grad.copy_(gradient)

    #     for index, param in enumerate(self._module_parameters):
    #         self._accum_grad_hooks.append(
    #             param.register_post_accumulate_grad_hook(
    #                 functools.partial(
    #                     compiled_accum_grad_hook,
    #                     param_index=index,
    #                 )
    #             )
    #         )
