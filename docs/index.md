---
title: Home
nav_order: 1
---

# ABM Uni v3 Documentation

Documentation for an Agent Based Model (ABM) simulating Uniswap v3, with equations rendered via MathJax (inline `$...$`, display `$$...$$`).

## Pages

{% assign nav_pages = site.html_pages | sort: "nav_order" %}
<ul>
  {% for p in nav_pages %}
    {% if p.nav_order and p.title %}
      {% unless p.nav_exclude %}
        {% if p.url != page.url %}
          {% assign href = p.url %}
          {% if href == "/" %}
            {% assign href = "index.html" %}
          {% else %}
            {% assign href = href | remove_first: "/" %}
          {% endif %}
          <li><a href="{{ href }}">{{ p.title | default: p.name }}</a></li>
        {% endif %}
      {% endunless %}
    {% endif %}
  {% endfor %}
</ul>
