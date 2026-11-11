"""Manage output from a single experiment run."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def result_directory(output_dir, *, keep):
    """Hide live output from Git and discard failed or non-formal runs."""
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError(f'Choose a real output directory: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f'Choose an empty output directory: {output_dir}')

    guard = output_dir / '.gitignore'
    with guard.open('x', encoding='utf-8') as stream:
        stream.write('*\n')
    if any(path != guard for path in output_dir.iterdir()):
        guard.unlink()
        raise ValueError(f'Choose an empty output directory: {output_dir}')

    completed = False
    try:
        yield
        completed = True
    finally:
        if completed and keep:
            guard.unlink()
        else:
            # Move it out of the results path before deleting the ignore file.
            with TemporaryDirectory(prefix='.discarded-run-', dir=output_dir.parent) as trash:
                output_dir.rename(Path(trash) / 'output')
