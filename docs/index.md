---
title: Home
nav_order: 1
---

# ABM Uni v3 Documentation

Documentation for an Agent Based Model (ABM) simulating Uniswap v3, with equations rendered via MathJax (inline `$...$`, display `$$...$$`).

## Pages

{% assign nav_pages = site.pages | where_exp: "p", "p.nav_exclude != true and p.title" | sort: "nav_order" %}
<ul>
  {% for p in nav_pages %}
    {% if p.url != page.url %}
      {% assign href = p.url %}
      {% if href == "/" %}
        {% assign href = "index.html" %}
      {% else %}
        {% assign href = href | remove_first: "/" %}
      {% endif %}
      <li><a href="{{ href }}">{{ p.title | default: p.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>
