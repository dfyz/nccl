import argparse
import json


def main(args):
    with open(args.input_trace_file) as f:
        trace = json.load(f)

    trace_events = trace['traceEvents']
    nccl_timings = trace['ncclTimings']

    nccl_events = []
    for ev in trace_events:
        if ev.get('ph') == 'X':
            name = ev.get('name')
            kernel_type = None
            if 'AllGather' in name:
                kernel_type = 'ag'
            elif 'MixedPrecisionReduceScatter' in name:
               # TODO: handle regular ReduceScatter, too
               kernel_type = 'rs'
            if kernel_type is not None:
                nccl_events.append((ev['ts'], kernel_type))
    nccl_events.sort(key=lambda x: x[0])

    new_events = []
    nccl_channel_start = 4141
    used_channels = set()
    recv_tid, send_tid = 0, 1

    for ev_idx in range(len(nccl_events)):
        # The PyTorch profiler ignores the very first collective for some reason, while NCCL doesn't, so we start from the last event.
        trace_start_us, needed_type = nccl_events[-1 - ev_idx]
        for ch_idx, ch_data in enumerate(nccl_timings):
            if args.only_channel and ch_idx not in args.only_channel:
                continue
            if ev_idx >= len(ch_data):
                continue
            kernel_data = ch_data[-1 - ev_idx]
            assert kernel_data[0] == needed_type, (ev_idx, kernel_data[0], needed_type)
            local_step = 0
            timing_count = 8
            last_recv_ts_us, last_send_ts_us = 0, 0
            used_channels.add(ch_idx)
            for ii in range(1, len(kernel_data), timing_count):
                step_recv, step_send, size, \
                    wait_recv_ts_ns, wait_send_ts_ns, post_recv_ts_ns, post_send_ts_ns, post_send_before_barrier_ts_ns \
                        = kernel_data[ii:ii+timing_count]
                wait_recv_ts_us, wait_send_ts_us, post_recv_ts_us, post_send_ts_us, post_send_before_barrier_ts_us = map(
                    lambda x: int(round(x / 1000)),
                    (wait_recv_ts_ns, wait_send_ts_ns, post_recv_ts_ns, post_send_ts_ns, post_send_before_barrier_ts_ns)
                )

                def add(start, finish, name, args, tid):
                    if start < finish:
                        new_events.append({
                            'name': name,
                            'ph': 'X',
                            'ts': trace_start_us + start,
                            'dur': finish - start,
                            'pid': nccl_channel_start + ch_idx,
                            'tid': tid,
                            'args': args,
                        })

                if step_recv:
                    def add_recv(start, finish, name, args):
                        add(start, finish, name, args, recv_tid)

                    add_recv(last_recv_ts_us, wait_recv_ts_us, 'Network', {})
                    process_start_us = wait_recv_ts_us
                    if step_send and wait_send_ts_us > wait_recv_ts_us:
                        add_recv(wait_recv_ts_us, wait_send_ts_us, 'Idle', {})
                        process_start_us = wait_send_ts_us
                    process_end_us = post_send_before_barrier_ts_us if step_send else post_recv_ts_us
                    add_recv(process_start_us, process_end_us, 'Process', {
                        'size': size,
                        'step': step_recv,
                        'local_step': local_step,
                    })
                    last_recv_ts_us = process_end_us

                if step_send:
                    def add_send(start, finish, name, args):
                        add(start, finish, name, args, send_tid)

                    add_send(last_send_ts_us, wait_send_ts_us, 'Network', {})
                    process_start_us = wait_send_ts_us
                    if step_recv and wait_recv_ts_us > wait_send_ts_us:
                        add_send(wait_send_ts_us, wait_recv_ts_us, 'Idle', {})
                        process_start_us = wait_recv_ts_us
                    add_send(process_start_us, post_send_before_barrier_ts_us, 'Process', {
                        'args': {
                            'size': size,
                            'step': step_send,
                            'local_step': local_step,
                        }
                    })
                    add_send(post_send_before_barrier_ts_us, post_send_ts_us, 'Membar', {})
                    last_send_ts_us = post_send_before_barrier_ts_us

                local_step += 1

    for ch_idx in used_channels:
        new_events.append({
            'name': 'process_name',
            'ph': 'M',
            'pid': nccl_channel_start + ch_idx,
            'args': {
                'name': f'Channel #{ch_idx:02}',
            },
        })
        new_events.append({
            'name': 'thread_name',
            'ph': 'M',
            'pid': nccl_channel_start + ch_idx,
            'tid': recv_tid,
            'args': {
                'name': f'Recv',
            },
        })
        new_events.append({
            'name': 'thread_name',
            'ph': 'M',
            'pid': nccl_channel_start + ch_idx,
            'tid': send_tid,
            'args': {
                'name': f'Send',
            },
        })

    with open(args.output_trace_file, 'w') as f:
        json.dump({
            'traceEvents': trace_events + new_events,
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_trace_file')
    parser.add_argument('output_trace_file')
    parser.add_argument('--only-channel', type=int, action='append')
    args = parser.parse_args()
    main(args)
