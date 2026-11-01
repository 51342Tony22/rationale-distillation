"""Versioned dispatch retaining signed production/training/evaluation cores."""
import argparse
from transfer128_replay_order_v3 import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('production','engineering','train','public'))
    parser.add_argument('action')
    parser.add_argument('--condition', choices=('M1','NLL','M3','M7','M6'))
    args = parser.parse_args()
    install()
    if args.stage == 'production':
        import transfer128_production_v1 as m
        actions = {'freeze-configuration':m.freeze_configuration, 'run':m.run,
                   'replay':m.replay, 'finalize':m.finalize}
        if args.action not in actions: parser.error('Unknown production action')
        result = actions[args.action]()
    elif args.stage == 'engineering':
        if args.action != 'if-supported': parser.error('Expected if-supported')
        from transfer128_production_v1 import load_configuration
        if load_configuration()['scientific_trial_enabled']:
            from preflight_transfer128_training_v1 import main as engineering
            result = engineering()
        else:
            result = {'status':'unsupported_no_expansion','GPU_engineering_executed':False}
    elif args.stage == 'train':
        if args.action != 'advance' or args.condition not in ('M7','M6'):
            parser.error('Expected advance --condition M7/M6')
        from train_transfer128_v1 import advance
        result = advance(args.condition)
    else:
        import public_evaluation_workflow_v1 as m
        if args.action == 'prepare': result = m.prepare()
        elif args.action == 'finalize': result = m.finalize()
        elif args.action == 'run' and args.condition: result = m.run(args.condition)
        else: parser.error('Expected prepare/finalize/run --condition')
    print({k:v for k,v in result.items() if k in
           ('schema','status','complete','conditions','selected_configuration','scientific_trial_enabled',
            'actual_trial_roots','device','supervised_tokens_including_eos')}, flush=True)


if __name__ == '__main__':
    main()
