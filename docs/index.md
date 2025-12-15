---
title: Home
nav_order: 1
---

# ABM Uni v3 Documentation

This site is built with **GitHub Pages (Jekyll)** and renders math with **MathJax**, so inline `$...$` and display `$$...$$` equations work in all notes.

## Pages

{% assign nav_pages = site.pages | where_exp: "p", "p.nav_exclude != true" | sort: "nav_order" %}
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
