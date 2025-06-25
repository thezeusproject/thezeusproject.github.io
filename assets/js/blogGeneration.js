import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dirBlog = path.resolve(__dirname, '../../content/blog');
const outputPath = path.resolve(__dirname, '../../blogs.json');

const blogs = [];

fs.readdirSync(dirBlog)
  .filter(file => file.endsWith('.md'))
  .forEach(file => {
    const filePath = path.join(dirBlog, file);
    const content = fs.readFileSync(filePath, 'utf8');
    const { data } = matter(content);

    blogs.push({
      id: file.replace('.md', ''),
      title: data.title,
      subtitle: data.subtitle,
    });
  });

fs.writeFileSync(outputPath, JSON.stringify(blogs, null, 2));
console.log(`blogs.json generated at ${outputPath}`);
