# Angular Cheat Sheet

Quick reference for Angular development.

## Component Decorators

```typescript
@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['./example.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [MyService]
})
```

## Lifecycle Hooks

```typescript
ngOnInit() { }           // After component init
ngOnDestroy() { }        // Before component destroy
ngOnChanges(changes: SimpleChanges) { }  // Input changes
ngAfterViewInit() { }    // After view init
ngDoCheck() { }          // Custom change detection
```

## Data Binding

```html
<!-- Interpolation -->
{{ value }}

<!-- Property Binding -->
<img [src]="imageUrl" [alt]="imageAlt">

<!-- Event Binding -->
<button (click)="onClick()">Click</button>

<!-- Two-way Binding -->
<input [(ngModel)]="name">

<!-- Template Reference -->
<input #nameInput>
<button (click)="nameInput.focus()">Focus</button>
```

## Structural Directives

```html
<!-- Conditional -->
<div *ngIf="condition">Visible if true</div>
<div *ngIf="condition; else elseBlock">Content</div>
<ng-template #elseBlock>Else content</ng-template>

<!-- Loop -->
<div *ngFor="let item of items; let i = index; trackBy: trackByFn">
  {{ i }}: {{ item.name }}
</div>

<!-- Switch -->
<div [ngSwitch]="value">
  <div *ngSwitchCase="'case1'">Case 1</div>
  <div *ngSwitchDefault>Default</div>
</div>
```

## Attribute Directives

```html
<!-- Dynamic Classes -->
<div [ngClass]="{'active': isActive, 'disabled': isDisabled}">
<div [ngClass]="getClasses()">

<!-- Dynamic Styles -->
<div [ngStyle]="{'color': textColor, 'font-size': fontSize + 'px'}">
<div [ngStyle]="getStyles()">

<!-- Custom Attribute -->
<div [appHighlight]="'yellow'">Highlighted</div>
```

## Pipes

```html
<!-- Built-in Pipes -->
{{ date | date:'short' }}
{{ price | currency:'USD' }}
{{ text | uppercase }}
{{ items | slice:0:5 }}

<!-- Custom Pipe -->
{{ value | customPipe:param1:param2 }}
```

## Forms

### Template-Driven
```html
<form #form="ngForm" (ngSubmit)="onSubmit()">
  <input name="name" [(ngModel)]="user.name" required>
  <input name="email" [(ngModel)]="user.email" required email>
  <button type="submit" [disabled]="!form.valid">Submit</button>
</form>
```

### Reactive Forms
```typescript
// Component
userForm = this.fb.group({
  name: ['', [Validators.required, Validators.minLength(2)]],
  email: ['', [Validators.required, Validators.email]],
  age: [null, [Validators.min(18)]]
});

// Template
<form [formGroup]="userForm" (ngSubmit)="onSubmit()">
  <input formControlName="name">
  <div *ngIf="userForm.get('name')?.invalid && userForm.get('name')?.touched">
    Error message
  </div>
</form>
```

## Services

```typescript
@Injectable({
  providedIn: 'root'
})
export class MyService {
  private data = new BehaviorSubject<any[]>([]);
  data$ = this.data.asObservable();

  getData(): Observable<any[]> {
    return this.http.get<any[]>('/api/data');
  }

  updateData(newData: any[]) {
    this.data.next(newData);
  }
}
```

## HTTP

```typescript
// GET
getUsers(): Observable<User[]> {
  return this.http.get<User[]>('/api/users');
}

// POST
createUser(user: User): Observable<User> {
  return this.http.post<User>('/api/users', user);
}

// PUT
updateUser(id: number, user: User): Observable<User> {
  return this.http.put<User>(`/api/users/${id}`, user);
}

// DELETE
deleteUser(id: number): Observable<void> {
  return this.http.delete<void>(`/api/users/${id}`);
}

// With Headers
getProtectedData(): Observable<any> {
  const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
  return this.http.get('/api/protected', { headers });
}
```

## Routing

```typescript
// Route Configuration
const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'users/:id', component: UserComponent },
  { path: 'admin', component: AdminComponent, canActivate: [AuthGuard] },
  { path: '**', redirectTo: '' }
];

// Navigation
this.router.navigate(['/users', userId]);
this.router.navigate(['/users', userId], { queryParams: { tab: 'profile' } });

// Get Route Parameters
constructor(private route: ActivatedRoute) {
  this.route.params.subscribe(params => {
    this.userId = params['id'];
  });
  
  this.route.queryParams.subscribe(params => {
    this.tab = params['tab'];
  });
}
```

## Guards

```typescript
@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  canActivate(): boolean {
    return this.authService.isAuthenticated();
  }
}

@Injectable({
  providedIn: 'root'
})
export class ResolveGuard implements Resolve<User> {
  resolve(): Observable<User> {
    return this.userService.getCurrentUser();
  }
}
```

## Interceptors

```typescript
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = localStorage.getItem('token');
    
    if (token) {
      req = req.clone({
        setHeaders: { Authorization: `Bearer ${token}` }
      });
    }
    
    return next.handle(req);
  }
}
```

## Testing

```typescript
// Component Test
describe('MyComponent', () => {
  let component: MyComponent;
  let fixture: ComponentFixture<MyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [MyComponent],
      providers: [MyService]
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(MyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

// Service Test
describe('MyService', () => {
  let service: MyService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [MyService]
    });
    service = TestBed.inject(MyService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  it('should fetch data', () => {
    service.getData().subscribe(data => {
      expect(data).toEqual(mockData);
    });

    const req = httpMock.expectOne('/api/data');
    req.flush(mockData);
  });
});
```

## Common CLI Commands

```bash
# Generate
ng generate component my-component
ng generate service my-service
ng generate pipe my-pipe
ng generate guard my-guard
ng generate module my-module

# Build & Serve
ng serve
ng build
ng build --prod
ng test
ng e2e

# Linting
ng lint
ng lint --fix
```

## Performance Tips

```typescript
// OnPush Change Detection
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush
})

// TrackBy Function
trackByFn(index: number, item: any): any {
  return item.id;
}

// Pure Pipe
@Pipe({
  name: 'myPipe',
  pure: true
})

// Lazy Loading
const routes: Routes = [
  {
    path: 'feature',
    loadChildren: () => import('./feature/feature.module').then(m => m.FeatureModule)
  }
];
```

## Error Handling

```typescript
// HTTP Error Handling
getData(): Observable<any> {
  return this.http.get('/api/data').pipe(
    catchError(error => {
      console.error('Error:', error);
      return throwError(() => new Error('Failed to fetch data'));
    })
  );
}

// Global Error Handler
@Injectable()
export class GlobalErrorHandler implements ErrorHandler {
  handleError(error: Error) {
    console.error('An error occurred:', error);
    // Log to service, show user notification, etc.
  }
}
```

## Environment Configuration

```typescript
// environment.ts
export const environment = {
  production: false,
  apiUrl: 'http://localhost:3000/api'
};

// environment.prod.ts
export const environment = {
  production: true,
  apiUrl: 'https://api.production.com'
};
```

## Common Patterns

```typescript
// Singleton Service
@Injectable({
  providedIn: 'root'
})

// Factory Provider
providers: [
  {
    provide: MyService,
    useFactory: (http: HttpClient) => new MyService(http),
    deps: [HttpClient]
  }
]

// Value Provider
providers: [
  { provide: 'API_URL', useValue: 'https://api.example.com' }
]
```
