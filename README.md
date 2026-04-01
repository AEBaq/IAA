# IAA

Recupération des labos

```bash
# 1. Mettre à jour les remotes
git checkout main
git pull origin main
git fetch gitlab

# 2. Mettre à jour la branche d’intégration avec mon travail
git checkout integration_gitlab
git merge main
# (résoudre conflits éventuels, puis `git add` + `git commit`)

# 3. Intégrer les nouveautés du squelette (GitLab)
git merge gitlab/master
# (résoudre conflits, puis `git add` + `git commit`)

# 4. Tester, puis fusionner vers main
git checkout main
git merge integration_gitlab
git push origin main
```

[template du lab1](https://github.com/duckietown/template-ros)
