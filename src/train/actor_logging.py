"""Optional epoch-level W&B logging, embedded in the standalone actor notebook."""
import json
from pathlib import Path


def start_actor_wandb(options, *, training_resume, saved_run, output_dir, config):
    if not options or not options.get('enabled', True):
        return None
    import wandb
    mode = options.get('mode', 'auto')
    if mode not in ('auto', 'new', 'resume'):
        raise ValueError('WANDB_MODE must be auto, new, or resume')
    explicit_id = options.get('run_id') or None
    project = options.get('project', 'primary_actor')
    entity = options.get('entity') or None
    previous = saved_run
    sidecar = Path(output_dir) / 'wandb_run.json'
    if previous is None and training_resume and sidecar.exists():
        previous = json.loads(sidecar.read_text())
    should_resume = mode == 'resume' or (mode == 'auto' and training_resume and (explicit_id or previous))
    if should_resume and not training_resume:
        raise ValueError('Do not append scratch training to an old W&B run; use WANDB_MODE="new"')
    run_id = explicit_id
    if should_resume and not run_id:
        if not previous:
            raise ValueError('Set WANDB_RUN_ID to resume an existing W&B run')
        if previous['project'] != project or (entity and previous['entity'] != entity):
            raise ValueError('Saved W&B project/entity differs; choose WANDB_MODE="new" or an explicit run ID')
        run_id = previous['id']
        entity = previous['entity']
    if wandb.run is not None:
        raise RuntimeError('An active W&B run already exists. Finish it before starting training again.')
    # Read the Colab secret only at runtime. Never put the key in config/checkpoints.
    try:
        from google.colab import userdata
    except ImportError:
        # Local CLI: use `wandb login` or WANDB_API_KEY; never persist the key here.
        wandb.login()
    else:
        try:
            token = userdata.get('WANDB_TOKEN')
        except Exception as exc:
            raise RuntimeError('Enable access to WANDB_TOKEN in Colab, or disable W&B') from exc
        if not token:
            raise ValueError('WANDB_TOKEN is empty')
        wandb.login(key=token, relogin=True)
        del token
    run = wandb.init(project=project, entity=entity, name=options.get('name', 'primary_actor'),
                     id=run_id, resume='must' if should_resume else 'never',
                     config=config, mode='online', save_code=False, sync_tensorboard=False)
    try:
        run.define_metric('epoch')
        run.define_metric('*', step_metric='epoch')
        identity = dict(id=run.id, project=run.project, entity=run.entity, name=run.name)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        pending = sidecar.with_suffix('.json.tmp')
        pending.write_text(json.dumps(identity, indent=2))
        pending.replace(sidecar)
        print(f'W&B run: {run.url} (id={run.id})')
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def log_actor_epoch(run, epoch, results, best_epoch, best_val_loss, learning_rate):
    if run is None:
        return
    metrics = {'epoch': epoch, 'optimization/learning_rate': learning_rate}
    for split in ('train', 'val'):
        for suffix in ('loss', 'prediction_loss', 'sigreg_loss', 'actor_nll', 'actor_mse'):
            metrics[f'{split}/{suffix}'] = results[f'{split}_{suffix}'][-1]
    for key in ('train_grad_norm', 'encoder_bn_gamma_rms', 'encoder_bn_running_var_min',
                'encoder_bn_running_var_max'):
        metrics[key] = results[key][-1]
    # Custom epoch axis, not explicit W&B step: continuing an existing run never
    # tries to write to a past internal step after a retry or logging fork.
    run.log(metrics)
    run.summary['best_epoch'] = best_epoch
    run.summary['best_validation_total'] = best_val_loss
