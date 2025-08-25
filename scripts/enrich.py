# scripts/enrich.py
import os
import json
import requests
import re
from math import ceil

api_key     = os.environ['TMDB_API_KEY']
base_search = 'https://api.themoviedb.org/3/search/movie'
base_movie  = 'https://api.themoviedb.org/3/movie'

def fetch_details(movie_id, lang='es-MX'):
    det_resp = requests.get(f"{base_movie}/{movie_id}", params={
        'api_key': api_key,
        'language': lang
    })
    det_resp.raise_for_status()
    det = det_resp.json()

    imgs_resp = requests.get(f"{base_movie}/{movie_id}/images", params={
        'api_key': api_key
    })
    imgs_resp.raise_for_status()
    imgs = imgs_resp.json()
    return det, imgs

def enrich_item(item):
    sample     = item['samples'][0]
    orig_title = sample.get('name','').strip()
    orig_year  = sample.get('anio','').strip()

    # 1) buscar en inglés con año
    params = {'api_key': api_key, 'language': 'en-US', 'query': orig_title}
    if orig_year:
        params['year'] = orig_year
    results = requests.get(base_search, params=params).json().get('results', [])

    # 2) si no hay, sin año
    if not results:
        params.pop('year', None)
        results = requests.get(base_search, params=params).json().get('results', [])

    if results:
        movie = results[0]
        tmdb_year = movie.get('release_date', '')[:4]
        # corregir año ±1
        if tmdb_year and orig_year and abs(int(tmdb_year) - int(orig_year)) == 1:
            corrected_year = tmdb_year
        else:
            corrected_year = orig_year or tmdb_year

        det, imgs = fetch_details(movie['id'], lang='es-MX')
        title_es   = det.get('title') or orig_title
        genres     = [g['name'] for g in det.get('genres', [])]
        overview   = det.get('overview', '').strip()
        release_date = det.get('release_date', '')
        final_year = release_date[:4] if release_date else corrected_year
        runtime    = det.get('runtime', 0)

        poster   = (f"https://image.tmdb.org/t/p/original{det.get('poster_path','')}"
                    if det.get('poster_path') else '')
        backdrop = (f"https://image.tmdb.org/t/p/original{det.get('backdrop_path','')}"
                    if det.get('backdrop_path') else '')
        logos    = imgs.get('logos', [])
        logo_url = (f"https://image.tmdb.org/t/p/original{logos[0]['file_path']}"
                    if logos else '')

        # unir géneros en cadena
        genero_str = ', '.join(genres)

        return {
            'name': genres[0] if genres else '',
            'samples': [{
                'name': title_es,
                'url': sample.get('url',''),
                'icono': poster,
                'iconoHorizontal': backdrop,
                'iconpng': logo_url,
                'type': 'PELICULA',
                'descripcion': overview,
                'anio': final_year,
                'genero': genero_str,
                'duracion': f"{runtime} min"
            }]
        }
    else:
        # no encontrado
        return {
            'name': 'Variado',
            'samples': [{
                'name': orig_title,
                'url': sample.get('url',''),
                'icono': '',
                'iconoHorizontal': '',
                'iconpng': '',
                'type': 'PELICULA',
                'descripcion': '',
                'anio': orig_year,
                'genero': '',
                'duracion': ''
            }]
        }

def main():
    with open('playlist.json', encoding='utf-8') as f:
        data = json.load(f)

    enriched = []
    for item in data:
        try:
            enriched.append(enrich_item(item))
        except Exception:
            enriched.append({
                'name': 'Variado',
                'samples': item.get('samples', [])
            })

    # dividir en dos mitades
    half = ceil(len(enriched) / 2)
    part1 = enriched[:half]
    part2 = enriched[half:]

    with open('playlist.enriched.part1.json', 'w', encoding='utf-8') as f:
        json.dump(part1, f, indent=2, ensure_ascii=False)
    with open('playlist.enriched.part2.json', 'w', encoding='utf-8') as f:
        json.dump(part2, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
