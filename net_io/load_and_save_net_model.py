from torch.nn import DataParallel
import os
import torch


def netio_load(net, state_dict, strict):
    keys = state_dict.keys()
    isparallel = all(['module' in k for k in keys])

    if isinstance(net, DataParallel):
        if isparallel:
            net.load_state_dict(state_dict, strict)
        else:
            net.module.load_state_dict(state_dict, strict)
    else:
        if isparallel:
            new_state_dict = OrderedDict()
            for k, v in model.items():
                name = k[7:]
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict,strict)
        else:
            net.load_state_dict(state_dict,strict)
    return net

def netio_save(net,epoch, save_dir,args):

    if isinstance(net, DataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    # state_dict = net.module.state_dict()

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict,
        'args': args},
        os.path.join(save_dir, '%04d.ckpt'%(epoch)))


def netio_trace(model, config):
    shape = [config.crop_size,]*3
    channel = config.input_channel
    sample_data = torch.rand(1,channel, shape[0],shape[1],shape[2]).cuda()
    model = model.eval()
    with torch.no_grad():
        trace = torch.jit.trace(model, sample_data)
        weight_file = config.resume
        trace_file = weight_file.replace('.ckpt','.trace')
        torch.jit.save(trace, trace_file)
        print('save model to ', trace_file)


