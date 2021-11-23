I used to think deploying my blog with github actions is unnecessary, but now... (真香)
Deploying with github actions is so much faster.

<!--more-->

## Introduction

I have two repositories: [sky-bro/blog-src](https://github.com/sky-bro/blog-src) and [sky-bro/sky-bro.github.io](https://github.com/sky-bro/sky-bro.github.io).
Before settign up github actions, I wrote my posts in [sky-bro/blog-src](https://github.com/sky-bro/blog-src),  then build it on my computer and push the built files (in the `public/` folder) to [sky-bro/sky-bro.github.io](https://github.com/sky-bro/sky-bro.github.io) using a script `./deploy.sh`.
Now every time I push to `blog-src`, the workflow runs, and deploy the newest build to the `sky-bro.github.io` repo.

## final workflow file

Put this in `.github/workflows/<workflow name>.yml`, mine is at `.github/workflows/gh-pages.yml`

```yml
name: deploy to sky-bro.github.io

on:
  push:
    branches:
      - master  # Set a branch to deploy
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-18.04
    steps:
      - uses: actions/checkout@v2
        with:
          submodules: true  # Fetch Hugo themes (true OR recursive)
          fetch-depth: 0    # Fetch all history for .GitInfo and .Lastmod

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: 'latest'
          extended: true

      - uses: actions/cache@v2
        with:
          path: /tmp/hugo_cache
          key: ${{ runner.os }}-hugomod-${{ hashFiles('**/go.sum') }}
          restore-keys: |
            ${{ runner.os }}-hugomod-

      - name: Build
        run: hugo --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }}
          external_repository: sky-bro/sky-bro.github.io
          publish_branch: master
          publish_dir: ./public
          full_commit_message: ${{ github.event.head_commit.message }}
```

## workflow explained

there are three main sections:

* `name`: name of the workflow (`deploy to sky-bro.github.io`)
* `on`: rules to trigger this workflow (my workflow will be triggered after pushing to the master branch or manually in the **Actions** tab)
* `jobs`: actual jobs to do after being trigered

This workflow only contains one job, and the name of this job is `deploy`, it has some steps, which will be executed in the given order.

### Step 1. check out files

Just copy.
The hugo themes are in `theme/` folder as submodules.
The fetch depth is set to 0 so we can fetch all history for `.GitInfo` and `.Lastmod`. (some hugo themes need this)

### Step 2. Set up hugo

Set the version number (`'0.79.1'`, `latest`), and decide whether or not you want to use the hugo extended. (If your theme uses `SASS/SCSS`, you need hugo extended)

### Step 3. Set hugo cache

optional, just copy.

### Step 4. Build your site

Giving `--minify` will let hugo minify any supported output format (HTML, XML etc.).

### Step 5. Deploy your site

This is the most important one.

* `deploy_key`, `github_token` or `personal_token`: three types of tokens, choose the one you need
* `publish_dir` is where the site files are generated, hugo uses `./public` by default
* `commit_message` or `full_commit_message`, `full_commit_message` will not include hash at the end of the deploy commit. `${{ github.event.head_commit.message }}` is the latest commit message of your source.
* `external_repository`, `publish_branch`: where you'll deploy you site (if using another repository)

If you have your site deployed in the same repository.

You can use `github_token`, just add `github_token: ${{ secrets.GITHUB_TOKEN }}` under `with:`, like below. This is the simplest. Then follow [First Deployment with GITHUB_TOKEN](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-first-deployment-with-github_token)

```yml
- name: Deploy
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./public
```

But sometimes, you would like to deploy your site to a different repository, you can only use `deploy_key` or `personal_token` to deploy to a different repository.

I used a `personal_token` (ref: [Create SSH Deploy Key](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-create-ssh-deploy-key)).

First generate your deploy key with the following command.

```shell
ssh-keygen -t rsa -b 4096 -C "$(git config user.email)" -f gh-pages -N ""
# You will get 2 files:
#   gh-pages.pub (public key)
#   gh-pages     (private key)
```

Next, go to the repository settings

* of your target repository (where you deploy you site to). Go to **Deploy Keys** and add your public key with the **Allow write access**.
* of your source repository. Go to **Secrets** and add your private key as **ACTIONS_DEPLOY_KEY**

## Refs

* [peaceiris/actions-hugo](https://github.com/peaceiris/actions-hugo)
* [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages), most contents of this post can be found here.
