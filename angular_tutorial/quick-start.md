# Angular Quick Start Guide

Get up and running with Angular in 15 minutes!

## 1. Install Prerequisites (2 min)

```bash
# Install Node.js from https://nodejs.org/ (LTS version)
# Verify installation
node --version
npm --version

# Install Angular CLI globally
npm install -g @angular/cli
```

## 2. Create Your First App (3 min)

```bash
# Create new project
ng new my-first-app
# Choose: Yes for routing, CSS for styling

# Navigate to project
cd my-first-app

# Start development server
ng serve
```

Open `http://localhost:4200` in your browser!

## 3. Understand the Structure (2 min)

```
src/
├── app/
│   ├── app.component.ts      # Main component
│   ├── app.component.html    # Main template
│   ├── app.component.css     # Main styles
│   ├── app.module.ts         # Main module
│   └── app-routing.module.ts # Routing
├── assets/                   # Images, icons, etc.
├── index.html               # Main HTML file
└── main.ts                  # Entry point
```

## 4. Create Your First Component (3 min)

```bash
# Generate a new component
ng generate component hello-world
# or shorthand: ng g c hello-world
```

Edit the generated files:

**hello-world.component.ts:**
```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'app-hello-world',
  templateUrl: './hello-world.component.html',
  styleUrls: ['./hello-world.component.css']
})
export class HelloWorldComponent {
  message = 'Hello from Angular!';
  count = 0;

  increment() {
    this.count++;
  }
}
```

**hello-world.component.html:**
```html
<div class="hello-world">
  <h2>{{ message }}</h2>
  <p>Count: {{ count }}</p>
  <button (click)="increment()">Increment</button>
</div>
```

**hello-world.component.css:**
```css
.hello-world {
  text-align: center;
  padding: 20px;
}

button {
  background: #007bff;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
}

button:hover {
  background: #0056b3;
}
```

## 5. Use Your Component (2 min)

Edit `app.component.html`:

```html
<app-hello-world></app-hello-world>
```

## 6. Add Data Binding (3 min)

Update `hello-world.component.ts`:

```typescript
export class HelloWorldComponent {
  message = 'Hello from Angular!';
  count = 0;
  inputText = '';

  increment() {
    this.count++;
  }

  updateMessage() {
    this.message = this.inputText || 'Hello from Angular!';
  }
}
```

Update `hello-world.component.html`:

```html
<div class="hello-world">
  <h2>{{ message }}</h2>
  <p>Count: {{ count }}</p>
  
  <div class="input-section">
    <input [(ngModel)]="inputText" placeholder="Enter new message">
    <button (click)="updateMessage()">Update Message</button>
  </div>
  
  <button (click)="increment()">Increment</button>
</div>
```

## 7. Add Routing (2 min)

Create a new component:
```bash
ng g c about
```

Update `app-routing.module.ts`:
```typescript
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HelloWorldComponent } from './hello-world/hello-world.component';
import { AboutComponent } from './about/about.component';

const routes: Routes = [
  { path: '', component: HelloWorldComponent },
  { path: 'about', component: AboutComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
```

Update `app.component.html`:
```html
<nav>
  <a routerLink="/">Home</a>
  <a routerLink="/about">About</a>
</nav>

<router-outlet></router-outlet>
```

## 8. Test Your App (1 min)

```bash
# Run tests
ng test

# Build for production
ng build --prod
```

## What You've Built

✅ A working Angular application  
✅ Component with data binding  
✅ Event handling  
✅ Two-way data binding  
✅ Basic routing  
✅ Responsive styling  

## Next Steps

1. **Learn More Concepts:**
   - Read the comprehensive tutorial
   - Study the cheat sheet
   - Build the example project

2. **Add Features:**
   - Forms (template and reactive)
   - HTTP services
   - Authentication
   - State management

3. **Best Practices:**
   - Follow Angular style guide
   - Write tests
   - Use TypeScript effectively
   - Implement proper error handling

## Common Issues & Solutions

**Issue: Component not showing**
- Check if component is declared in `app.module.ts`
- Verify selector in template

**Issue: Property binding not working**
- Check syntax: `[property]="value"`
- Verify component property exists

**Issue: Event not firing**
- Check syntax: `(event)="method()"`
- Verify method exists in component

**Issue: Routing not working**
- Check if `RouterModule` is imported
- Verify `<router-outlet>` is in template

## Quick Commands Reference

```bash
# Generate
ng g c component-name
ng g s service-name
ng g p pipe-name

# Serve & Build
ng serve
ng build
ng test

# Help
ng help
ng generate --help
```

## Congratulations! 🎉

You've successfully created your first Angular application! You now understand:
- Component structure
- Data binding
- Event handling
- Basic routing
- Angular CLI usage

Continue learning with the comprehensive tutorial and build more complex applications!
