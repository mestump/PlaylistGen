# PlaylistGen

PlaylistGen is a CLI tool that analyzes your Spotify listening history and local iTunes library,
enriches tracks with genre and mood tags, and generates curated playlists tailored to your taste profile.

## Installation

```bash
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
```

## Usage

```bash
# Generate playlists using default settings
playlistgen
```

```bash
# Force re-cache of all moods (rebuild Last.fm mood cache)
playlistgen recache-moods
```

Or via module:

```bash
python -m playlistgen
```

## Configuration
Create a `config.yml` file in your project root or `~/.playlistgen/config.yml` to override default settings
(see `playlistgen/config.yml` for defaults).

## Development

Run tests:

```bash
pytest
```
