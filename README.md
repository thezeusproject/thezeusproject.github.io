# Zeus Project

Welcome to the Zeus Project! This repository is a collection of educational posts and associated projects related to various topics in machine learning and data science. We aim to provide clear theoretical explanations along with hands-on projects to solidify understanding.

We welcome contributions from the community! If you have a topic you'd like to write about, please follow the steps below.

## How to Contribute

To contribute a new blog post and project, you'll need to add two files: one for the blog content and one for the project description.

### Step 1: Fork and Clone the Repository

Start by forking the repository to your own GitHub account and then clone it to your local machine.

```bash
git clone https://github.com/<your-username>/thezeusproject.github.io
cd thezeusproject.github.io
```
> Replace `<your-username>` with your GitHub username

### Step 2: Create New Content Files

A complete contribution consists of a blog post, a corresponding project, and an HTML page to display them.

1.  **Create a blog post**: Navigate to the `content/blog` directory. The easiest way to start is to copy an existing blog post file, for example `linear-regression.md`, and rename it to your new topic (e.g., `your-topic-name.md`).

2.  **Create a project file**: Similarly, go to `content/projects` and create a corresponding markdown file for the project tasks (e.g., `your-topic-name.md`). You can also copy an existing file from this directory to use as a template.

3.  **Create an HTML page**: Go to `pages/blog-pages`. Copy an existing HTML file (e.g., `linear-regression.html`) and rename it to match your topic (e.g., `your-topic-name.html`).

### Step 3: Add Your Content

Now, open the files you've created and add your content.

#### Blog Post (`content/blog/your-topic-name.md`)

At the beginning of your blog post file, you need to include a frontmatter section. This is crucial for the website to correctly display your post.

Here is an example of what the frontmatter should look like:

```markdown
<!-- filler line (always add this filler line or code wont work)-->
---
title: Your Blog Post Title
subtitle: A brief and engaging subtitle for your post.
author: your-github-username
---

![image](assets/images/your-awesome-image.png)

## Section 1: Introduction

Your amazing content starts here...
```

-   **`title`**: The title of your blog post.
-   **`subtitle`**: A short description that will appear under the title.
-   **`author`**: Your name or username.
-   **Image**: You can add a header image. **All images used in your blog post or project must be uploaded to the `/assets/images/` directory.** Link to them as shown above (e.g., `![image](assets/images/your-awesome-image.png)`).

After the frontmatter, you can write the full content of your blog post in Markdown.

#### Project File (`content/projects/your-topic-name.md`)

The project file should contain a list of tasks for a hands-on project related to your blog post. Use Markdown checklists to create the tasks.

Here is an example structure:

```markdown
### Project: Your Project Title

- [ ] **Task 1: A description of the first task**
  - [ ] A sub-task if needed.
- [ ] **Task 2: Another task**
```

### Step 4: Update the HTML file

Open the new HTML file you created (`pages/blog-pages/your-topic-name.html`) and make sure the title and the paths to the markdown files are correct.

- **Update Title**: Change the `<title>` tag to reflect your blog post's title.
- **Update Content Paths**: Locate the `fetch` calls in the JavaScript section at the bottom of the file. Update the paths to point to your new markdown files in `content/blog` and `content/projects`.

For example, if your topic is `your-topic-name`, the `fetch` calls should look like this:

```javascript
// ...
Promise.all([
    fetch('../../content/blog/your-topic-name.md').then(res => res.text()),
    fetch('../../content/projects/your-topic-name.md').then(res => res.text())
])
//...
```

### Step 5 : Manually update the `/assets/curriculum.js` file

Please update the file with your blog details under the const `blogs`.

### Step 6: Submit a Pull Request

Once you have created your blog post and project files, commit the changes and push them to your fork.

```bash
git add .
git commit -m "feat: Add new blog on [Your Topic]"
git push origin main
```

Then, go to the original repository on GitHub and open a Pull Request. We will review your contribution and merge it if everything looks good.

## 🏛️ Hall of Projects - Community Showcase

The Zeus Project features a **Hall of Projects** where community members can showcase their machine learning projects! This creates a collaborative space for learners to share their work and discover inspiring projects from others.

### How to Submit Your Project

Anyone can submit their ML project to be featured in our Hall of Projects. Here's how:

#### Method 1: Direct JSON Submission (Recommended)

1. **Fork the Repository**: Fork this repository to your GitHub account

2. **Edit projects.json**: Add your project to the `projects.json` file in the root directory

3. **Project Entry Format**:
```json
{
  "id": "your-project-id",
  "title": "Your Project Title",
  "author": "your-github-username",
  "githubRepo": "https://github.com/yourusername/your-repo",
  "description": "Brief description of your project (2-3 sentences max)",
  "tags": ["tag1", "tag2", "tag3"],
  "category": "classical-ml",
  "thumbnail": "assets/images/your-thumbnail.png",
  "difficulty": "beginner",
  "dateSubmitted": "2024-12-25",
  "upvotes": 0
}
```

4. **Field Guidelines**:
   - **id**: Unique identifier (lowercase, hyphens for spaces)
   - **title**: Clear, descriptive project name
   - **author**: Your GitHub username or display name
   - **githubRepo**: Link to your project repository or demo
   - **description**: 2-3 sentences explaining what your project does
   - **tags**: 3-5 relevant keywords (lowercase, hyphens for spaces)
   - **category**: Choose from: `classical-ml`, `deep-learning`, `nlp`, `computer-vision`, `reinforcement-learning`, `data-science`
   - **thumbnail**: Upload image to `/assets/images/` and reference the path
   - **difficulty**: Choose from: `beginner`, `intermediate`, `advanced`
   - **dateSubmitted**: Submission date in YYYY-MM-DD format
   - **upvotes**: Always start with 0

5. **Add Thumbnail** (Optional but Recommended):
   - Upload a 400x250px image to `/assets/images/`
   - Use formats: PNG, JPG, or GIF
   - Name it descriptively (e.g., `neural-network-visualization.png`)

6. **Submit Pull Request**:
```bash
git add .
git commit -m "feat: Add [Your Project Name] to Hall of Projects"
git push origin main
```

Then create a Pull Request with the title: **"Add [Your Project Name] to Hall of Projects"**

#### Method 2: Full Tutorial Submission

If you want to create a complete tutorial (like our existing content), follow the original contribution guidelines above, then also add your project to `projects.json` linking to your new tutorial page.

### Project Guidelines

✅ **What Makes a Good Submission**:
- Clear, working implementation
- Good documentation or README
- Relevant to machine learning/data science
- Educational value for the community
- Original work or significant improvements to existing projects

✅ **Categories We Accept**:
- **Classical ML**: Linear/logistic regression, SVM, decision trees, clustering
- **Deep Learning**: Neural networks, CNNs, RNNs, transformers
- **NLP**: Text processing, sentiment analysis, language models
- **Computer Vision**: Image classification, object detection, image generation
- **Reinforcement Learning**: Game AI, robot control, optimization
- **Data Science**: Analysis, visualization, statistical modeling

### Community Features

The Hall of Projects includes:
- **🔍 Smart Search**: Find projects by title, author, or tags
- **🏷️ Category Filters**: Browse by ML domain and difficulty level
- **👍 Community Upvotes**: Show appreciation for great projects
- **📱 Mobile-Friendly**: Works perfectly on all devices
- **📊 Live Stats**: Track community growth and engagement

### Questions?

If you need help with your submission or have questions about the Hall of Projects, feel free to:
- Open an issue with the `hall-of-projects` label
- Check existing projects in `projects.json` for examples
- Review the [projects showcase page](https://thezeusproject.github.io/projects.html)

Let's build an amazing community of ML learners together! 🚀

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Thank you for contributing to the Zeus Project! 
