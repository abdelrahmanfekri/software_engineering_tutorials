# Angular Tutorial - Quick & Comprehensive

A complete guide to Angular development covering all essential concepts with practical examples.

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Core Concepts](#core-concepts)
3. [Components](#components)
4. [Templates & Data Binding](#templates--data-binding)
5. [Services & Dependency Injection](#services--dependency-injection)
6. [Routing](#routing)
7. [Forms](#forms)
8. [HTTP & APIs](#http--apis)
9. [State Management](#state-management)
10. [Testing](#testing)
11. [Deployment](#deployment)

## Setup & Installation

### Prerequisites
- Node.js (v18+)
- npm or yarn

### Install Angular CLI
```bash
npm install -g @angular/cli
```

### Create New Project
```bash
ng new my-angular-app
cd my-angular-app
ng serve
```

## Core Concepts

### Angular Architecture
- **Components**: UI building blocks
- **Services**: Business logic and data
- **Modules**: Feature organization
- **Dependency Injection**: Service management

### Key Files
- `main.ts`: Application entry point
- `app.module.ts`: Root module
- `app.component.ts`: Root component
- `angular.json`: Build configuration

## Components

### Component Structure
```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['./example.component.css']
})
export class ExampleComponent {
  title = 'My Component';
  
  onButtonClick() {
    console.log('Button clicked!');
  }
}
```

### Component Lifecycle
- `ngOnInit`: Component initialization
- `ngOnDestroy`: Component cleanup
- `ngOnChanges`: Input property changes
- `ngAfterViewInit`: View initialization

### Input/Output
```typescript
@Input() data: string;
@Output() dataChange = new EventEmitter<string>();

sendData() {
  this.dataChange.emit('New data');
}
```

## Templates & Data Binding

### Interpolation
```html
<h1>{{ title }}</h1>
<p>Count: {{ count }}</p>
```

### Property Binding
```html
<img [src]="imageUrl" [alt]="imageAlt">
<button [disabled]="isDisabled">Click me</button>
```

### Event Binding
```html
<button (click)="onClick()">Click</button>
<input (input)="onInput($event)" (blur)="onBlur()">
```

### Two-Way Binding
```html
<input [(ngModel)]="name" placeholder="Enter name">
<p>Hello, {{ name }}!</p>
```

### Structural Directives
```html
<div *ngIf="isVisible">This is visible</div>

<ul>
  <li *ngFor="let item of items; let i = index">
    {{ i + 1 }}. {{ item.name }}
  </li>
</ul>

<div [ngSwitch]="status">
  <p *ngSwitchCase="'active'">Active</p>
  <p *ngSwitchCase="'inactive'">Inactive</p>
  <p *ngSwitchDefault>Unknown</p>
</div>
```

### Attribute Directives
```html
<div [ngClass]="{'active': isActive, 'disabled': isDisabled}">
  Dynamic classes
</div>

<div [ngStyle]="{'color': textColor, 'font-size': fontSize + 'px'}">
  Dynamic styles
</div>
```

## Services & Dependency Injection

### Service Creation
```typescript
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private data: any[] = [];

  getData() {
    return this.data;
  }

  addData(item: any) {
    this.data.push(item);
  }
}
```

### Service Usage
```typescript
import { Component } from '@angular/core';
import { DataService } from './data.service';

@Component({
  selector: 'app-example',
  template: '<div>{{ data | json }}</div>'
})
export class ExampleComponent {
  data: any[];

  constructor(private dataService: DataService) {
    this.data = this.dataService.getData();
  }
}
```

### HTTP Service
```typescript
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'https://api.example.com';

  constructor(private http: HttpClient) {}

  getUsers(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/users`);
  }

  createUser(user: any): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/users`, user);
  }
}
```

## Routing

### Route Configuration
```typescript
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { AboutComponent } from './about/about.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'about', component: AboutComponent },
  { path: 'users/:id', component: UserComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
```

### Navigation
```html
<nav>
  <a routerLink="/">Home</a>
  <a routerLink="/about">About</a>
  <a [routerLink]="['/users', userId]">User Profile</a>
</nav>

<router-outlet></router-outlet>
```

### Programmatic Navigation
```typescript
import { Router } from '@angular/router';

constructor(private router: Router) {}

navigateToAbout() {
  this.router.navigate(['/about']);
}

navigateWithParams() {
  this.router.navigate(['/users', 123]);
}
```

### Route Guards
```typescript
import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(private router: Router) {}

  canActivate(): boolean {
    if (this.isAuthenticated()) {
      return true;
    }
    this.router.navigate(['/login']);
    return false;
  }

  private isAuthenticated(): boolean {
    return !!localStorage.getItem('token');
  }
}
```

## Forms

### Template-Driven Forms
```html
<form #userForm="ngForm" (ngSubmit)="onSubmit()">
  <input name="name" [(ngModel)]="user.name" required>
  <input name="email" [(ngModel)]="user.email" required email>
  <button type="submit" [disabled]="!userForm.valid">Submit</button>
</form>
```

### Reactive Forms
```typescript
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

export class UserFormComponent {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {
    this.userForm = this.fb.group({
      name: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      age: [null, [Validators.min(18)]]
    });
  }

  onSubmit() {
    if (this.userForm.valid) {
      console.log(this.userForm.value);
    }
  }
}
```

```html
<form [formGroup]="userForm" (ngSubmit)="onSubmit()">
  <input formControlName="name" placeholder="Name">
  <div *ngIf="userForm.get('name')?.invalid && userForm.get('name')?.touched">
    Name is required and must be at least 2 characters
  </div>
  
  <input formControlName="email" placeholder="Email">
  <div *ngIf="userForm.get('email')?.invalid && userForm.get('email')?.touched">
    Please enter a valid email
  </div>
  
  <button type="submit" [disabled]="userForm.invalid">Submit</button>
</form>
```

## HTTP & APIs

### HTTP Interceptor
```typescript
import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = localStorage.getItem('token');
    
    if (token) {
      req = req.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`
        }
      });
    }
    
    return next.handle(req);
  }
}
```

### Error Handling
```typescript
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

getUsers(): Observable<any[]> {
  return this.http.get<any[]>(`${this.apiUrl}/users`).pipe(
    catchError(error => {
      console.error('Error fetching users:', error);
      return throwError(() => new Error('Failed to fetch users'));
    })
  );
}
```

## State Management

### Simple State with Services
```typescript
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StateService {
  private userSubject = new BehaviorSubject<any>(null);
  user$ = this.userSubject.asObservable();

  setUser(user: any) {
    this.userSubject.next(user);
  }

  getUser() {
    return this.userSubject.value;
  }
}
```

### NgRx (Advanced State Management)
```typescript
// Actions
export const loadUsers = createAction('[Users] Load Users');
export const loadUsersSuccess = createAction('[Users] Load Users Success', props<{users: any[]}>());

// Reducer
export const usersReducer = createReducer(
  initialState,
  on(loadUsers, state => ({ ...state, loading: true })),
  on(loadUsersSuccess, (state, { users }) => ({ ...state, users, loading: false }))
);

// Effects
@Injectable()
export class UsersEffects {
  loadUsers$ = createEffect(() => this.actions$.pipe(
    ofType(loadUsers),
    mergeMap(() => this.userService.getUsers().pipe(
      map(users => loadUsersSuccess({ users }))
    ))
  ));

  constructor(
    private actions$: Actions,
    private userService: UserService
  ) {}
}
```

## Testing

### Component Testing
```typescript
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ExampleComponent } from './example.component';

describe('ExampleComponent', () => {
  let component: ExampleComponent;
  let fixture: ComponentFixture<ExampleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExampleComponent ]
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ExampleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display title', () => {
    const compiled = fixture.nativeElement;
    expect(compiled.querySelector('h1').textContent).toContain('My Component');
  });
});
```

### Service Testing
```typescript
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiService } from './api.service';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ApiService]
    });
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch users', () => {
    const mockUsers = [{ id: 1, name: 'John' }];

    service.getUsers().subscribe(users => {
      expect(users).toEqual(mockUsers);
    });

    const req = httpMock.expectOne('https://api.example.com/users');
    expect(req.request.method).toBe('GET');
    req.flush(mockUsers);
  });
});
```

## Deployment

### Build for Production
```bash
ng build --prod
```

### Build Configuration
```json
{
  "configurations": {
    "production": {
      "optimization": true,
      "outputHashing": "all",
      "sourceMap": false,
      "extractCss": true,
      "namedChunks": false,
      "aot": true,
      "extractLicenses": true,
      "vendorChunk": false,
      "buildOptimizer": true
    }
  }
}
```

### Docker Deployment
```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist/my-angular-app /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Best Practices

### Performance
- Use `OnPush` change detection strategy
- Implement `trackBy` functions for `*ngFor`
- Lazy load modules and components
- Use pure pipes for transformations

### Code Organization
- Follow Angular style guide
- Use feature modules for organization
- Implement proper error handling
- Write comprehensive tests

### Security
- Sanitize user inputs
- Use Angular built-in XSS protection
- Implement proper authentication
- Validate all form inputs

## Quick Start Checklist

- [ ] Install Node.js and Angular CLI
- [ ] Create new Angular project
- [ ] Understand component structure
- [ ] Learn data binding concepts
- [ ] Create and use services
- [ ] Implement routing
- [ ] Build forms (template and reactive)
- [ ] Make HTTP requests
- [ ] Write basic tests
- [ ] Build and deploy

## Common Commands

```bash
# Generate components
ng generate component my-component
ng g c my-component

# Generate services
ng generate service my-service
ng g s my-service

# Generate modules
ng generate module my-module
ng g m my-module

# Generate guards
ng generate guard auth
ng g g auth

# Generate pipes
ng generate pipe my-pipe
ng g p my-pipe

# Build and serve
ng serve
ng build
ng test
ng e2e
```

## Resources

- [Angular Official Documentation](https://angular.io/docs)
- [Angular Style Guide](https://angular.io/guide/styleguide)
- [Angular CLI Reference](https://angular.io/cli)
- [Angular Material](https://material.angular.io/)
- [RxJS Documentation](https://rxjs.dev/)

---

This tutorial covers the essential Angular concepts you need to build production-ready applications. Practice with the examples and gradually build more complex features as you become comfortable with the basics.
