# Frontend Guide - Pfizer EMR Alert System

## Overview

The frontend of the Pfizer EMR Alert System is a modern web application that provides a clinical decision support interface for healthcare professionals. It integrates with the backend API to deliver AI-powered patient analysis and Drug A prescription recommendations.

## Architecture

### Technology Stack
- **Frontend Server**: Python HTTP server (`emr_ui_server.py`)
- **UI Framework**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with Pfizer brand colors
- **Icons**: Font Awesome 6.0
- **API Communication**: Fetch API with CORS support

### Directory Structure
```
frontend/
├── server/
│   ├── __init__.py
│   └── emr_ui_server.py          # UI server implementation
├── static/
│   ├── css/
│   │   ├── main.css             # Main stylesheet
│   │   └── clinical-decision-support.css  # Clinical UI components
│   ├── js/
│   │   ├── config.js            # Configuration management
│   │   ├── main.js              # Core application logic
│   │   └── clinical-decision-support.js  # Clinical decision support
│   ├── images/
│   │   └── logo.png             # Pfizer logo
│   └── fonts/                   # Custom fonts
└── templates/
    ├── index.html               # Main application template
    ├── layouts/
    │   └── base.html            # Base layout template
    └── components/
        └── alert-panel.html     # Alert panel component
```

## Core Components

### 1. UI Server (`emr_ui_server.py`)

The UI server is a Python HTTP server that serves the frontend application with the following features:

- **CORS Support**: Enables cross-origin requests to the API
- **Health Check Integration**: Waits for API server to be ready
- **Static File Serving**: Serves CSS, JS, and image files
- **Template Rendering**: Serves HTML templates
- **Error Handling**: Graceful error handling and port management

**Key Features:**
- Automatic API health checking before startup
- Port conflict resolution (tries alternative ports)
- Custom request handler with CORS headers
- Template path resolution for clean URLs

### 2. Main Application (`index.html`)

The main HTML template provides the complete user interface:

**Header Section:**
- Pfizer branding and system identification
- Real-time system status indicator
- Version information

**Sidebar:**
- Patient list with search functionality
- Patient status indicators (alerts, recommendations)
- Dynamic patient loading from API

**Main Content Area:**
- Patient information display
- Multi-step patient form (4 steps)
- AI analysis results panel
- Clinical decision support components

**Key UI Elements:**
- Responsive design with mobile navigation
- Loading overlays and progress indicators
- Form validation and step navigation
- Alert panels with detailed patient information

### 3. JavaScript Architecture

#### Configuration (`config.js`)
Centralized configuration management:
- API endpoints and settings
- UI behavior configuration
- Error messages and success notifications
- Feature flags and environment-specific settings
- Local storage keys for data persistence

#### Main Application (`main.js`)
Core application functionality:
- **Patient Management**: CRUD operations for patients
- **AI Analysis**: Integration with prediction API
- **Form Handling**: Multi-step form with validation
- **API Communication**: Robust error handling and retries
- **UI State Management**: Dynamic content updates
- **Health Monitoring**: Periodic API health checks

#### Clinical Decision Support (`clinical-decision-support.js`)
Advanced clinical features:
- **Eligibility Assessment**: Drug A treatment eligibility
- **Risk Visualization**: Interactive risk factor displays
- **Decision Flow**: Step-by-step clinical guidance
- **Contraindication Analysis**: Safety assessment tools

### 4. Styling System (`main.css`)

Comprehensive CSS framework with:

**Design System:**
- Pfizer brand color palette
- Medical system color scheme
- Consistent spacing and typography
- Responsive grid system

**Component Styles:**
- Form components with validation states
- Alert panels with status indicators
- Patient cards with priority sorting
- Clinical decision support components

**Responsive Design:**
- Mobile-first approach
- Flexible layouts for different screen sizes
- Touch-friendly interface elements

## Key Features

### 1. Patient Management

**Patient List:**
- Real-time patient data loading
- Search and filtering capabilities
- Priority sorting (Drug A recommendations first)
- Status indicators for alerts and recommendations

**Patient Form:**
- 4-step multi-step form process
- Real-time validation
- Auto-save functionality
- Comprehensive patient data collection

**Form Steps:**
1. **Basic Information**: Name, age, gender, physician ID, dates
2. **Medical History**: Location, insurance, contraindication level
3. **Symptoms & Comorbidities**: Checkbox selections for conditions
4. **Review & Submit**: Data validation and submission

### 2. AI Analysis Integration

**Prediction API Integration:**
- Real-time AI analysis requests
- Patient data transformation for API format
- Results visualization and interpretation
- Confidence level display

**Analysis Results:**
- Drug A prescription probability
- Risk factor identification
- Clinical eligibility assessment
- Alert recommendations

### 3. Clinical Decision Support

**Eligibility Assessment:**
- Age eligibility verification
- 5-day treatment window validation
- High-risk patient identification
- Contraindication level assessment

**Risk Visualization:**
- Interactive risk factor displays
- Visual risk scoring
- Factor importance indicators
- Clinical context presentation

**Decision Support Flow:**
- Step-by-step clinical guidance
- Evidence-based recommendations
- Safety considerations
- Treatment pathway visualization

### 4. User Experience Features

**Real-time Updates:**
- Live API health monitoring
- Dynamic content updates
- Real-time form validation
- Instant feedback on actions

**Error Handling:**
- Comprehensive error messages
- Retry mechanisms for failed requests
- Graceful degradation
- User-friendly error displays

**Accessibility:**
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Touch-friendly interface

## API Integration

### Endpoints Used

**Health Check:**
- `GET /health` - System health verification

**Patient Management:**
- `GET /patients` - Retrieve patient list
- `POST /patients` - Create new patient
- `GET /patients/{id}` - Get specific patient

**AI Analysis:**
- `POST /predict` - Run AI prediction analysis
- `GET /model/info` - Get model information
- `GET /model/features` - Get feature information

### Data Flow

1. **Patient Creation**: Form data → API validation → Database storage
2. **AI Analysis**: Patient data → Feature engineering → Model prediction → Results display
3. **Real-time Updates**: API polling → Data refresh → UI updates

### Error Handling

- Network error detection and retry logic
- API timeout handling
- Validation error display
- Graceful fallback mechanisms

## Development Guidelines

### Code Organization

**JavaScript Modules:**
- Configuration management in `config.js`
- Core functionality in `main.js`
- Clinical features in `clinical-decision-support.js`
- Global functions and utilities

**CSS Architecture:**
- CSS custom properties for theming
- Component-based styling
- Responsive design patterns
- Accessibility considerations

### Best Practices

**Performance:**
- Lazy loading of non-critical components
- Debounced API calls
- Efficient DOM manipulation
- Minimal re-renders

**Security:**
- Input validation and sanitization
- CORS configuration
- XSS prevention
- Secure API communication

**Maintainability:**
- Modular code structure
- Comprehensive error handling
- Consistent naming conventions
- Detailed logging and debugging

## Deployment

### Local Development

1. **Start the UI Server:**
   ```bash
   cd frontend
   python server/emr_ui_server.py
   ```

2. **Access the Application:**
   - URL: `http://localhost:8080`
   - API Server: `http://localhost:8000`

### Production Deployment

**Docker Support:**
- Dockerfile for containerized deployment
- Docker Compose for multi-service setup
- Health check integration
- Environment variable configuration

**Configuration:**
- Environment-specific API URLs
- Feature flag management
- Error reporting configuration
- Performance monitoring setup

## Troubleshooting

### Common Issues

**API Connection Problems:**
- Check API server status
- Verify CORS configuration
- Check network connectivity
- Review error logs

**Form Validation Issues:**
- Check required field validation
- Verify data format requirements
- Review error messages
- Test form submission flow

**UI Rendering Problems:**
- Check CSS file loading
- Verify JavaScript execution
- Review browser console errors
- Test responsive design

### Debug Tools

**Browser Developer Tools:**
- Console logging for debugging
- Network tab for API monitoring
- Elements tab for DOM inspection
- Performance profiling

**Application Debugging:**
- Built-in debug functions
- Error tracking and reporting
- Performance monitoring
- User interaction logging

## Future Enhancements

### Planned Features

**Advanced Analytics:**
- Patient outcome tracking
- Treatment effectiveness metrics
- Clinical decision analytics
- Performance dashboards

**Enhanced UI/UX:**
- Dark mode support
- Advanced filtering options
- Bulk operations
- Export functionality

**Integration Capabilities:**
- EMR system integration
- Third-party API connections
- Data synchronization
- Real-time notifications

### Technical Improvements

**Performance Optimization:**
- Code splitting and lazy loading
- Caching strategies
- Bundle optimization
- Progressive web app features

**Accessibility Enhancements:**
- WCAG 2.1 compliance
- Voice navigation support
- High contrast themes
- Screen reader optimization

---

## Quick Start

1. **Prerequisites:**
   - Python 3.7+
   - Backend API server running
   - Modern web browser

2. **Start the Frontend:**
   ```bash
   cd frontend
   python server/emr_ui_server.py
   ```

3. **Access the Application:**
   - Open `http://localhost:8080` in your browser
   - The system will automatically check API connectivity

4. **Basic Usage:**
   - Add patients using the "Add Patient" button
   - Select patients from the sidebar
   - Run AI analysis using the "AI Analysis" button
   - Review clinical decision support recommendations

For detailed API documentation, see the [API Guide](../backend/api/API_GUIDE.md).

For backend architecture details, see the [Backend Guide](../backend/BACKEND_GUIDE.md).
