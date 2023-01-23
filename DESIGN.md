# App loading service

## BOSS App Analysis manager

Micro-front module in *BOSS* serves operators to compare results of loading time of various mobile / web app using multiple variables (location, device OS and types, etc...),
using an AI layer that will allow training the loading / loaded expected objects per app, it will allow to analyze based on video recorded, regardless whether it was done by automation or employee of Playtika or external person,
Results will be available in the *BOSS App Analysis manager* as well as *Grafana*

### List all apps

- Filter by name and deleted
- Sorted by name, the latest duration
- Available data in list – name, created, last test date, last test duration
- Permission per app?
- Operations – Add, edit, disable, test (not available for deleted), view metrics
- Status icons:

| Icon                       | Color | Annotation | Schedule | Last duration (threshold) |
|----------------------------|-------|------------|----------|---------------------------|
| mdi-alert-decagram         | Gray  | Partial    | *        | *                         | 
| mdi-alert-decagram-outline | Gray  | -          | *        | *                         | 
| mdi-speedometer            | Gray  | +          | -        | Below or equal            | 
| mdi-speedometer-slow       | Gray  | +          | -        | Above                     | 
| mdi-speedometer            | Gray  | +          | +        | Below or equal            | 
| mdi-speedometer-slow       | Gray  | +          | +        | Above                     | 

###	Add / edit app
- Set name of the app
- Slow threshold in milliseconds
- Upload video to choose frames for annotation
- Annotate images (set zones and classify with label – loading / loaded)
- Publish (Upload and Train)

### Disable
- Turn off schedule

###	Trigger test 
- Set name of the test
- Define trigger:
  - Once 
  - Specific days on specific time of the day 
  - Specific dates / time of the day
- Types 
  - Recorded video upload (MVP)
  - Automation (integration to mobile farm)
  - Crowd testing (integration with providers) – later phase
- Metrics
  - Dashboard (hosted Grafana) with the ability to choose name of the app, time range, device type

## App Analysis Server:

Python YOLO v5 based image with custom web server to serve the micro-front of *BOSS App Analysis manager* and share statistics with *Prometheus* 

### Configuration

```yaml
WebSerer:
  Port: 

Kafka: 
  Host:
  Port:
  TopicPrefix:

Database:
  Engine: # MVP release only MySQL supported
  Host: # Hostname, IP or FQDN of MySQL database
  Port: 
  Database: 
  Username: 
  Password: 
```

### Volumes (and backed-up)

- Apps dataset
- Trained models
- Results

### Database

#### Tables

| Table                     | Description                                                                                                                                                               |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Apps                      | Holds all the apps                                                                                                                                                        |
| Datasets                  | Holds dataset items of specific app, uniqueness is based on the app Id and file name, content is a pointer to storage, will also contain create date and last update date | 
| Annotations               | Holds annotations of specific image file, contains Id, label, x, y, width and height                                                                                      |
| DetectionJobs             | Holds list of detection jobs, contains Id, name, status, timestamp and duration                                                                                           |
| DetectionJobEvents        | Holds events (predications) related to specific job execution, contains id, detection job id, timestamp, label, file name                                                 |
| PlanProviders             | Holds plan provider details, contains id, name, initiate endpoint, get recording endpoint                                                                                 |
| Plans                     | Holds plans details, contains Id, app name, plan provider id, status, days, dates, time of day                                                                            |
| PlanExecutionInstructions | Holds plan execution instructions, contains Id, plan id, device type, OS, location                                                                                        |

### Web Server

#### HTTP Status

| Status Code | Description                                                                                                                                 |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 200         | Ok, will return expected payload of the request                                                                                             |
| 400         | Bad request, will take place when the request body is invalid, body of response will hold parameter `error` with the reason for the failure | 
| 404         | Not found, request URI parameter was not found                                                                                              |

#### Endpoints

##### Get all apps

| Method | Endpoint  | Description                   |
|--------|-----------|-------------------------------|
| GET    | /api/apps | Get list of all app's details |

###### Response

```json5
[
    {  
      "dataset": [ //List of images and their metadata
        { 
          "fileName": "filename.ext",
          "content": "content", // Values: BASE64 value of the image, up to 1MB per image 
          "created": 123456, // Values: float represents the timestamp
          "annotations": [
            {
              "label": "name of the label", // Values: loading, loaded
              "x": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the left
              "y": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the top
              "width": 0.0, // Values: float, 0.0 - 1.0 as width percentage of the object
              "height": 0.0, // Values: float, 0.0 - 1.0 as height percentage of the object
            }
          ]
        }
      ]
    }
]
```

##### Get specific app details

| Method | Endpoint             | Description               |
|--------|----------------------|---------------------------|
| GET    | /api/apps/{app_name} | Get specific app details  |

###### Response

```json5
{  
  "dataset": [ //List of images and their metadata
    { 
      "fileName": "filename.ext",
      "content": "content", // Values: BASE64 value of the image, up to 1MB per image 
      "created": 123456, // Values: float represents the timestamp
      "annotations": [
        {
          "label": "name of the label", // Values: loading, loaded
          "x": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the left
          "y": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the top
          "width": 0.0, // Values: float, 0.0 - 1.0 as width percentage of the object
          "height": 0.0, // Values: float, 0.0 - 1.0 as height percentage of the object
        }
      ]
    }
  ]
}
```

##### Train app

| Method | Endpoint            | Description                    |
|--------|---------------------|--------------------------------|
| POST   | /api/apps/{appName} | Create and train new app model |

###### Request

```json5
{  
  "dataset": [ //List of images and their metadata
    { 
      "fileName": "filename.ext",
      "content": "content", // Values: BASE64 value of the image, up to 1MB per image 
      "created": 123456, // Values: float represents the timestamp
      "annotations": [
        {
          "label": "name of the label", // Values: loading, loaded
          "x": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the left
          "y": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the top
          "width": 0.0, // Values: float, 0.0 - 1.0 as width percentage of the object
          "height": 0.0, // Values: float, 0.0 - 1.0 as height percentage of the object
        }
      ]
    }
  ]
}
```

###### Response

```json5
{  
  "dataset": [ //List of images and their metadata
    { 
      "fileName": "filename.ext",
      "content": "content", // Values: BASE64 value of the image, up to 1MB per image 
      "created": 123456, // Values: float represents the timestamp
      "annotations": [
        {
          "label": "name of the label", // Values: loading, loaded
          "x": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the left
          "y": 0.0, // Values: float, 0.0 - 1.0 as percentage from center of the object compared to the top
          "width": 0.0, // Values: float, 0.0 - 1.0 as width percentage of the object
          "height": 0.0, // Values: float, 0.0 - 1.0 as height percentage of the object
        }
      ]
    }
  ]
}
```

##### Delete app

| Method | Endpoint            | Description                                     |
|--------|---------------------|-------------------------------------------------|
| DELETE | /api/apps/{appName} | Delete all files associated with a specific app |

##### Retrieve app detection

| Method | Endpoint                   | Description                       |
|--------|----------------------------|-----------------------------------|
| GET    | /api/apps/{appName}/detect | Retrieve details of app detection |

###### Response

```json5
[ 
  {
    "detectionId": "uuid", // Values: string, unique Id of detection instance
    "status": "processed", // Values: pending, processing, processed
    "timestamp": "timestamp", // Values: float, represents timestamp of the analysis start time
    "duration": "duration", // Values: float, represents time passed between first loading to first loaded events, if not loaded image identified, will be the length of the test file
  }
]

```

##### Retrieve detection details

| Method | Endpoint                                 | Description                                             |
|--------|------------------------------------------|---------------------------------------------------------|
| GET    | /api/apps/{appName}/detect/{detectionId} | Retrieve events related to specific job of specific app |

###### Response

```json5
{  
  "detectionId": "uuid",
  "status": "completed", // Values: pending, processing, processed, disabled
  "timestamp": "timestamp", // Values: float, represents timestamp of the analysis start time
  "duration": "duration", // Values: float, represents time passed between first loading to first loaded events, if not loaded image identified, will be the length of the test file
  "events": [ // List of events
    {
      "label": "loading", // Values: loading, loaded
      "fileName": "filename.ext",
      "timestamp": "timestamp", // Values: float, represents timestamp since beginning of the video
    }
  ],
  "metadata": { // Data provided by the requestor (e.g. location, device OS, device type, browser, ISP, etc...)   
    
  }
}
```

##### Detect app

| Method | Endpoint                   | Description                                                      |
|--------|----------------------------|------------------------------------------------------------------|
| POST   | /api/apps/{appName}/detect | Process data sent with specific model of the requested {appName} |

###### Request

```json5
{  
  "extension": "jpg", // Values: string represents the file extension for testing
  "content": "content", // Values: BASE64 value of the image or video to analyze
  "metadata": { // Data provided by the requestor (e.g. device location, device OS, device type, browser, ISP, etc...)   
    
  }
}
```

###### Response

```json5
{  
  "jobName": "name of the analysis job",
  "jobExecutionId": "uuid",
  "status": "completed", // Values: pending, processing, processed, disabled
}
```

##### Get app detection plans

| Method | Endpoint                        | Description                                        |
|--------|---------------------------------|----------------------------------------------------|
| POST   | /api/app/{appName}/detect/plans | Get all future plans for detection of specific app |

###### Response

```json5
[
  {
    "type": "automation", // Values: automation, crowd
    "metadata": { // Based on schedule type parameters to invoke the detection tool, 
      "devices": [ "Samsung S22 Android 13", "Samsung S22 Android 12" ],
      "regions": [ "US-NY", "CN", "IL" ]
    },
    "schedule": {
      "dates": ["2023-01-01", "2023-02-01"], // List of dates format YYYY-mm-DD, not working with days
      "days": ["Mon", "Thu"], // List of Values: Sun, Mon, Tue, Wed, Thu, Fri, Sat, not working with dates
      "time": ["00:00:00", "12:00:00"] // List of Time of the day format HH:MM:ss 
    }
  }
]
```

##### Create app detection plan

| Method | Endpoint                       | Description                                      |
|--------|--------------------------------|--------------------------------------------------|
| POST   | /api/app/{appName}/detect/plan | Create plan for future detection of specific app |

###### Request

```json5
{
  "type": "automation", // Values: automation, crowd
  "status": "pending", // Values: pending, disabled
  "metadata": { // Based on schedule type parameters to invoke the detection tool, 
    "devices": [ "Samsung S22 Android 13", "Samsung S22 Android 12" ],
    "regions": [ "US-NY", "CN", "IL" ]
  },
  "schedule": {
    "dates": ["2023-01-01", "2023-02-01"], // List of dates format YYYY-mm-DD, not working with days
    "days": ["Mon", "Thu"], // List of Values: Sun, Mon, Tue, Wed, Thu, Fri, Sat, not working with dates
    "time": ["00:00:00", "12:00:00"] // List of Time of the day format HH:MM:ss 
  }
}
```

###### Response

```json5
{  
  "planId": "uuid",
  "status": "pending", // Values: pending, processing, processed, disabled
}
```

##### Delete app detection plan

| Method | Endpoint                                | Description                                                 |
|--------|-----------------------------------------|-------------------------------------------------------------|
| DELETE | /api/app/{appName}/detect/plan/{planId} | Delete a specific future plan for detection of specific app |

###### Response

```json5
[
  {
    "type": "automation", // Values: automation, crowd
    "status": "disabled", // Values: pending, disabled
    "metadata": { // Based on schedule type parameters to invoke the detection tool, 
      "devices": [ "Samsung S22 Android 13", "Samsung S22 Android 12" ],
      "regions": [ "US-NY", "CN", "IL" ]
    },
    "schedule": {
      "dates": ["2023-01-01", "2023-02-01"], // List of dates format YYYY-mm-DD, not working with days
      "days": ["Mon", "Thu"], // List of Values: Sun, Mon, Tue, Wed, Thu, Fri, Sat, not working with dates
      "time": ["00:00:00", "12:00:00"] // List of Time of the day format HH:MM:ss 
    }
  }
]
```

##### Prometheus' metrics exporter

| Method | Endpoint | Description                                   |
|--------|----------|-----------------------------------------------|
| GET    | /metrics | Exposes metrics related to execution of jobs  |

```ini
# HELP app_analysis_job_events_count Counter of objects detected
# TYPE app_analysis_job_events_count gauge
app_analysis_job_events_count {appName="", jobName="", label="", fileName=""} 1

# HELP app_analysis_job_load_time Load time in milliseconds
# TYPE app_analysis_job_load_time gauge
app_analysis_job_load_time {appName="", jobName="", fileNameLoading="", fileNameLoaded="", status=""} 550

# HELP app_analysis_job_count Counter of jobs executed
# TYPE app_analysis_job_count gauge
app_analysis_job_count {appName="", jobName="", duration="", status=""} 1
``` 

#### Process

##### Annotation process

Annotation of an image is when labeling a specific zone as an object, 
each object detection algorithm expects different convention of the position of the label, 
YOLO expecting to get X-Center, Y-Center, Width and Height as percentages between 0.0 to 1.0.

```javascript
function getYOLOPixelInPositions(objectPositions) {
    const yoloPositionsInPixels = {
        xCenter: objectPositions.xLeft + ((objectPositions.xRight - objectPositions.xLeft) / 2),
        yCenter: objectPositions.yTop + ((objectPositions.yBottom - objectPositions.yTop) / 2),
        width: objectPositions.xRight - objectPositions.xLeft,
        height: objectPositions.yBottom - objectPositions.yTop 
    };
    
    return yoloPositionsInPixels;
}

function getYOLOPixelPositions(imageDimensions, objectPositions) {
    const yoloPositionsInPixels = getYOLOPixelInPositions(objectPositions);

    const yoloPositions = {
        xCenter: yoloPositionsInPixels.xCenter / imageDimensions.width,
        yCenter: yoloPositionsInPixels.yCenter / imageDimensions.height,
        width: yoloPositionsInPixels.width / imageDimensions.width,
        height: yoloPositionsInPixels.height / imageDimensions.height 
    };
    
    return yoloPositions;
}

function getYOLOAnnotation(labelId, imageDimensions, objectPositions) {
    const yoloPositions = getYOLOPixelPositions(imageDimensions, objectPositions);
    const annotationParams = [
        labelId,
        yoloPositions.xCenter,
        yoloPositions.yCenter,
        yoloPositions.width,
        yoloPositions.height
    ];

    const annotation =  annotationParams.join(" ");
    
    return annotation;
}

const labelId = 0; // Loading

const image = {
    width: 800,
    height: 1920
};

const objectPositions = {
    xLeft: 100,
    xRight: 500,
    yTop: 1200,
    yBottom: 1300
};

const annotation  = getYOLOAnnotation(labelId, image, objectPositions)
console.log(annotation);

/*
Result:
0 0.375 0.6510416666666666 0.5 0.052083333333333336
*/
```

##### Training process

###### Single Model

*Set up*
	
1. Create within datasets directory `playtika` dataset

2. Create the following directories
   - `/usr/src/datasets/playtika/images`
   - `/usr/src/datasets/playtika/images/train`
   - `/usr/src/datasets/playtika/images/val`
   - `/usr/src/datasets/playtika/labels`
   - `/usr/src/datasets/playtika/labels/train`
   - `/usr/src/datasets/playtika/labels/val`

3. Create YAML file name `playtika.yaml` in the root directory of dataset, content:
    ```yaml
    train: "/usr/src/datasets/playtika/images/train"
    val: "/usr/src/datasets/playtika/images/val"
    
    nc: 2
    names: ['loading', 'loaded']
    ```

*Execution*
	
1. Extract the app name from the request
   
    If `app_name` already exists, stop process and send failed status message
   
2. Save all images to both directories under `/usr/src/datasets/playtika/images/` (train, val)

3. Create text file per image named same as the image it represents, replace the extension to `txt`, in both directories under `/usr/src/datasets/playtika/labels` (train, val), content:
    ```text
    {label_id} {x} {y} {width} {height} 
    ```

4. Run process of `train.py` (as in-proc, not CLI)

    ```bash
    python train.py --epochs 1200 --data "/usr/src/datasets/playtika/playtika.yaml" --project "/usr/src/datasets/playtika/" --name="models" --exist-ok
   ```

###### Per App Model
	
1. Extract the app name from the request
   
    If `app_name` already exists, stop process and send failed status message

2. Follow steps of [Single Model for all apps](#single-model), instead of use the `playtika` name, use the `app_name`

3. Store result to DB

4. Publish message through *Kafka*, topic `{Kafka.TopicPrefix}.app.trained` with the same payload as `GET /api/apps/{app_name}` 

##### Plan execution

1. Scheduler service will identify which plans to execute

2. Per plan will extract the details for execution
    - app name (represent the model and images)
    - plan type
    - desired devices
    - desired locations to execute the plan

3. Send POST HTTP request tp `Plan.{planType}.Endpoints.Initiate`

4. Get execution ID and store it

5. Send GET HTTP request tp `Plan.{planType}.Endpoints.GetRecording` in intervals of 1 minute up until the status becomes 200 (Found)

6. Send file as base64 to [detection](#detect-app) with all metadata of execution returned from initiate call

7. Publish message through *Kafka*, topic `{Kafka.TopicPrefix}.plan.executed` with the same payload as returned from the initiate call and status of get recording 

##### Detect process

1. Save bytes to file name `/tmp/{app_name}.uuid.ext`

2. Run process of `detect.py` (as in-proc, not CLI)

    *Single model*
    ```bash
    python detect.py --data "/usr/src/datasets/playtika/playtika.yaml" --weights "/usr/src/datasets/playtika/models/weights/best.pt" --source "/tmp/$app_name.uuid.ext" --project "/usr/src/datasets/playtika" --name="result" --exist-ok
    ```

    *Model per app*
    ```bash
    python detect.py --data "/usr/src/datasets/$app_name/$app_name.yaml" --weights "/usr/src/datasets/$app_name/models/weights/best.pt" --source "/tmp/$app_name.uuid.ext" --project "/usr/src/datasets/$app_name" --name="result" --exist-ok
    ```

3. Store result to DB

4. Publish message through *Kafka*, topic `{Kafka.TopicPrefix}.job.processed` with the same payload as `GET /api/apps/{app_name}/analyze/{name}/{job_id}` 

## Open questions:
- Single model vs. model per app
- Versioning of jobs
- Upload a video file using email with mail relay (External users)
